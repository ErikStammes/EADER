""" Wraps around a model and extracts its attention map (CAM or Grad-CAM) """
from collections import defaultdict

import torch
import torch.nn.functional as F

def attention_map_wrapper(model):
    """ Wrapper that dynamically inherits from the given argument model"""
    class AttentionMapWrapper(model):
        """ Wrapper that stores the forward features and gradients to create an attention map """
        def __init__(self, model_params, features_layer, weights_layer=None, attention_type='gcam'):
            super(AttentionMapWrapper, self).__init__(**model_params)
            self.features_layer = features_layer
            self.type = attention_type

            if attention_type == 'cam' and weights_layer is None:
                raise ValueError('When using CAM you need to specify a weights layer')
            self.weights_layer = weights_layer

            self.img_shape = None
            self.forward_features = {}
            self.backward_features = {}

            self.register_hooks()

        def one_hot_encode(self, tensor, shape):
            """ One hot encode a tensor """
            tensor = tensor.view(-1, 1)
            one_hot = torch.zeros(shape, device=tensor.device)
            one_hot.scatter_(1, tensor, 1.0)
            return one_hot

        def register_hooks(self):
            """ Register forward and backward hooks that store features and gradients from the given layer """
            def forward_hook(_module, _forward_input, forward_output):
                self.forward_features[forward_output.device] = forward_output

            def backward_hook(_module, _backward_input, backward_output):
                self.backward_features[backward_output[0].device] = backward_output[0]

            for name, module in self.named_modules():
                if name == self.features_layer:
                    module.register_forward_hook(forward_hook)
                    module.register_backward_hook(backward_hook)

        def get_target_label_mapping(self, labels):
            """
            labels: the indices at which the target labels occur.
            Returns a dictionary with as keys the indexes of each item in the batch and as values a list of targets.
            """
            label_indices = labels.nonzero()
            label_mapping = defaultdict(list)
            for i in range(label_indices.size(0)):
                label_mapping[label_indices[i, 0].item()].append(label_indices[i, 1].item())
            return label_mapping

        def compute_gradcam(self, device):
            """ Computes the GradCAM heatmaps """
            assert self.img_shape is not None, 'GradCAM can only be computed after the forward pass'
            weights = F.adaptive_avg_pool2d(self.backward_features[device], 1)
            gcam = torch.mul(self.forward_features[device], weights).sum(dim=1, keepdim=True)
            gcam = F.relu(gcam)
            gcam = F.interpolate(gcam, self.img_shape, mode='bilinear', align_corners=False)
            batch_size, channel_size, height, width = gcam.shape
            gcam = gcam.view(batch_size, -1)
            gcam -= gcam.min(dim=1, keepdim=True)[0]
            gcam_max = gcam.max(dim=1, keepdim=True)[0]
            gcam /= torch.where(gcam_max != 0, gcam_max, torch.ones_like(gcam_max))
            gcam = gcam.view(batch_size, channel_size, height, width)
            return gcam

        def forward(self, images, labels=None, compute_gradcam=True):
            """ Extracts the GradCAM after forwarding the input images """
            logits = super().forward(images)
            if not compute_gradcam:
                return logits
            self.img_shape = images.size()[2:]

            if labels is None:
                if self.training:
                    raise Exception('Labels are necessary during training to create Grad-CAMs')
                else:
                    labels = logits

            #TODO: improve this part
            if self.type == 'cam':
                _, feature_num_channels, feature_height, feature_width = self.forward_features[images.device].shape
                label_indices = labels.nonzero()
                selected_cams = []
                selected_targets = []
                original_targets = []
                selected_images = []
                weights = [weights for name, weights in self.named_parameters() if name == self.weights_layer][0]
                for j in range(label_indices.size(0)):
                    batch_idx = label_indices[j, 0]
                    target = label_indices[j, 1]
                    features = self.forward_features[images.device][batch_idx].reshape((feature_num_channels, feature_height * feature_width))
                    target_weights = weights[target]
                    cam = torch.matmul(target_weights, features)
                    cam = cam.reshape(feature_height, feature_width).unsqueeze(0).unsqueeze(0)
                    cam = F.interpolate(cam, self.img_shape, mode='bilinear', align_corners=False)
                    batch_size, channel_size, height, width = cam.shape
                    cam = cam.view(batch_size, -1)
                    cam -= cam.min(dim=1, keepdim=True)[0]
                    cam_max = cam.max(dim=1, keepdim=True)[0]
                    cam /= torch.where(cam_max != 0, cam_max, torch.ones_like(cam_max))
                    cam = cam.view(batch_size, channel_size, height, width)
                    selected_cams.append(cam.squeeze(0))
                    selected_targets.append(target)
                    original_targets.append(labels[batch_idx])
                    selected_images.append(images[batch_idx])
                cams = torch.stack(selected_cams)
                new_targets = torch.as_tensor(selected_targets, device=labels.device)
                original_targets = torch.stack(original_targets)
                new_images = torch.stack(selected_images)
                logits = torch.sigmoid(logits)
                return logits, (cams, new_images, new_targets, original_targets)
            elif self.type == 'gcam':
                label_mapping = self.get_target_label_mapping(labels)
                counts = [len(values) for values in label_mapping.values()]

                gcams = []
                new_images = []
                original_targets = []
                new_targets = []
                for i in range(max(counts)):
                    # Each iteration we select targets based on the label mapping
                    selected_indices = torch.zeros_like(labels, dtype=torch.float64, device=images.device)
                    targets = []
                    for j in range(len(label_mapping)):
                        if i < len(label_mapping[j]):
                            # Set targets (one-hot encoded)
                            selected_indices[j][label_mapping[j][i]] = 1
                            targets.append(label_mapping[j][i])
                        else:
                            # Save as -1 so we can recognize that this iteration didn't have any targets for this batch item
                            targets.append(-1)
                    targets = torch.as_tensor(targets, device=images.device)

                    # Do backward pass to compute the backward features
                    logits.backward(gradient=selected_indices, retain_graph=True)

                    gcam = self.compute_gradcam(images.device)

                    gcam = gcam[targets != -1]
                    og_images = images[targets != -1]
                    og_targets = labels[targets != -1]
                    targets = targets[targets != -1]
                    gcams.extend(gcam)
                    new_images.extend(og_images)
                    original_targets.extend(og_targets)
                    new_targets.extend(targets)

                gcams = torch.stack(gcams)
                new_images = torch.stack(new_images)
                original_targets = torch.stack(original_targets)
                new_targets = torch.as_tensor(new_targets, device=images.device)
                logits = torch.sigmoid(logits)
                return logits, (gcams, new_images, new_targets, original_targets)
            else:
                raise ValueError(f'Invalid attention type: {self.type}')
    return AttentionMapWrapper
