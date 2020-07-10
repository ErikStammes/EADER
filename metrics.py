import torch

from data.pascal_voc import LABELS

class Metrics():
    def __init__(self, num_classes, ignore_index=255, background_index=0):
        self.num_classes = num_classes + 1 # add background class
        self.ignore_index = ignore_index
        self.background_index = background_index
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes, dtype=torch.int64)
        self.union = torch.zeros(self.num_classes, dtype=torch.int64)
        self.gt_pixel_count = torch.zeros(self.num_classes, dtype=torch.int64)
        self.predicted_pixel_count = torch.zeros(self.num_classes, dtype=torch.int64)
        self.count = 0

    def update(self, prediction, target):
        """ Updates metrics based on the prediction and the target segmentation map """
        assert prediction.shape == target.shape
        target = target.view(-1)
        _prediction = prediction.clone().view(-1)
        _prediction[target == self.ignore_index] = self.ignore_index
        for class_idx in range(self.num_classes):
            pred_indices = _prediction == class_idx
            target_indices = target == class_idx
            if pred_indices.sum() == 0 and target_indices.sum() == 0:
                continue
            intersection = pred_indices[target_indices].sum()
            union = pred_indices.sum() + target_indices.sum() - intersection
            self.intersection[class_idx] += intersection
            self.union[class_idx] += union
            self.count += 1
            if class_idx == self.background_index:
                continue # skip the background class for precision and recall
            self.gt_pixel_count[class_idx] += target_indices.sum()
            self.predicted_pixel_count[class_idx] += pred_indices.sum()

    def miou(self, class_index=None):
        """ Returns the mean intersection over union computed over all pixels since init/reset
            If class_index is given (int), will return the mean intersection over union for that class.
        """
        if class_index:
            return self.intersection[class_index].float() / (self.union[class_index].float() + 1e-10)
        return (self.intersection.float() / (self.union.float() + 1e-10)).mean()

    def precision(self, class_index=None, skip_background=False):
        """ Returns the precision computed over all pixels since init/reset
            If class_index is given (int), will return the precision for that class.
        """
        if class_index:
            return self.intersection[class_index].float() / (self.predicted_pixel_count[class_index].float() + 1e-10)
        if skip_background:
            indices = torch.ones(self.num_classes, dtype=torch.uint8)
            indices[self.background_index] = 0
            return (self.intersection[indices].float() / (self.predicted_pixel_count[indices].float() + 1e-10)).mean()
        return (self.intersection.float() / (self.predicted_pixel_count.float() + 1e-10)).mean()

    def recall(self, class_index=None, skip_background=False):
        """ Returns the recall computed over all pixels since init/reset
            If class_index is given (int), will return the recall for that class.
        """
        if class_index:
            return self.intersection[class_index].float() / (self.gt_pixel_count[class_index].float() + 1e-10)
        if skip_background:
            indices = torch.ones(self.num_classes, dtype=torch.uint8)
            indices[self.background_index] = 0
            return (self.intersection[indices].float() / (self.gt_pixel_count[indices].float() + 1e-10)).mean()
        return (self.intersection.float() / (self.gt_pixel_count.float() + 1e-10)).mean()

    def print_scores_per_class(self):
        """ Prints the mean scores per class """
        labels = ['background', *LABELS]
        miou_string = 'Per class mIoU scores:\n'
        precision_string = 'Per class precision scores:\n'
        recall_string = 'Per class recall scores:\n'
        for class_index in range(self.num_classes):
            label_string = f'{labels[class_index]:<15}'
            miou_string += f'{label_string}: {self.miou(class_index).item():.3f}\n'
            if class_index == self.background_index:
                continue
            precision_string += f'{label_string}: {self.precision(class_index).item():.3f}\n'
            recall_string += f'{label_string}: {self.recall(class_index).item():.3f}\n'
        print(miou_string, precision_string, recall_string)

    def __str__(self):
        return f'mIoU: {self.miou():.3f}\nPrecision: {self.precision():.3f}\nRecall: {self.recall():.3f}'

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
