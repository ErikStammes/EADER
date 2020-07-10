""" Load localizer, classifier and shape model """
import torch

from torchvision import models as torchvision_models
from torch.hub import load_state_dict_from_url

from attention_map_wrapper import attention_map_wrapper

from .deeplab_largefov import VGG16_LargeFOV

MODEL_URLS = {
    'inception': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'wide_resnet50': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
}

def load_localizer(architecture, pretrained, attention_type, num_classes):
    """ Load the localizer model, which is wrapped with an attention map extractor """
    if architecture in ('resnet18', 'resnet101', 'resnet50', 'wide_resnet50', 'resnet101deeplab'):
        model = load_resnet_localizer(architecture, pretrained, attention_type, num_classes)
    elif architecture == 'inception':
        final_conv_layer = 'Mixed_7c'
        model = attention_map_wrapper(torchvision_models.Inception3)({}, final_conv_layer, attention_type)
        if state_dict is not None:
            state_dict = torch.load(state_dict)['state_dict']
            model.load_state_dict(state_dict)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS[architecture])
            model.load_state_dict(state_dict)
        model.aux_logits = False
        del model.AuxLogits
        model.fc = torch.nn.Linear(2048, num_classes)
    elif architecture == 'vgg16':
        final_conv_layer = 'features'
        layers = make_vgg_layers('D')
        model_args = {'features': layers}
        model = attention_map_wrapper(torchvision_models.VGG)(model_args, final_conv_layer, attention_type)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS[architecture])
            model.load_state_dict(state_dict)
        model.classifier[6] = torch.nn.Linear(4096, num_classes)
    elif architecture == 'deeplab_largefov':
        final_conv_layer = 'features'
        model = attention_map_wrapper(VGG16_LargeFOV)({}, final_conv_layer, attention_type)
        if pretrained:
            state_dict = load_state_dict_from_url(MODEL_URLS['vgg16'])
            model.load_state_dict(state_dict)
        model.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))
        model.classifier = torch.nn.Linear(512, num_classes)
    else:
        raise Exception(f'Invalid localizer model specified: {architecture}')
    return model

def load_resnet_localizer(architecture, pretrained, attention_type, num_classes):
    """ Util function for loading ResNet architectures as localizer """
    final_conv_layer = 'layer4'
    weights_layer = 'fc.weight'
    if architecture == 'resnet18':
        model_args = {'block': torchvision_models.resnet.BasicBlock, 'layers': [2, 2, 2, 2]}
    elif architecture in  ('resnet101', 'resnet101deeplab'):
        model_args = {'block': torchvision_models.resnet.Bottleneck, 'layers': [3, 4, 23, 3]}
        if architecture == 'resnet101deeplab':
            model_args['replace_stride_with_dilation'] = [False, True, True]
    elif architecture in ('resnet50', 'wide_resnet50'):
        model_args = {'block': torchvision_models.resnet.Bottleneck, 'layers': [3, 4, 6, 3]}
        if architecture == 'wide_resnet50':
            model_args['width_per_group'] = 128
    model = attention_map_wrapper(torchvision_models.ResNet)(model_args, final_conv_layer, weights_layer,
                                  attention_type)
    if pretrained:
        state_dict = load_state_dict_from_url(MODEL_URLS[architecture])
        model.load_state_dict(state_dict)
    if architecture == 'resnet18':
        model.fc = torch.nn.Linear(512, num_classes)
    else:
        model.fc = torch.nn.Linear(2048, num_classes)
    return model

def load_adversarial(architecture, pretrained, num_classes):
    """ Load the adversarial model """
    if architecture == 'resnet18':
        model = torchvision_models.resnet18(pretrained=pretrained)
        model.fc = torch.nn.Linear(512, num_classes)
    elif architecture == 'resnet101':
        model = torchvision_models.resnet101(pretrained=pretrained)
        model.fc = torch.nn.Linear(2048, num_classes)
    else:
        raise Exception(f'Invalid adversarial model specified: {architecture}')
    return model

def make_vgg_layers(cfg):
    cfgs = {
        'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }
    cfg = cfgs[cfg]
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [torch.nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = torch.nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, torch.nn.ReLU(inplace=True)]
            in_channels = v
    return torch.nn.Sequential(*layers)
