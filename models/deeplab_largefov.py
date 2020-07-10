""" Code adapted from
https://github.com/wangleihitcs/DeepLab-V1-PyTorch
"""

import torch
import torch.nn as nn

class VGG16_LargeFOV(nn.Module):
    def __init__(self, num_classes=21, init_weights=True):
        super(VGG16_LargeFOV, self).__init__()
        self.features = nn.Sequential(
            ### conv1_1 conv1_2 maxpooling
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ### conv2_1 conv2_2 maxpooling
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            ### conv3_1 conv3_2 conv3_3 maxpooling
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),


            ### conv4_1 conv4_2 conv4_3 maxpooling(stride=1)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            ### conv5_1 conv5_2 conv5_3 (dilated convolution dilation=2, padding=2)
            ### maxpooling(stride=1)
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ### average pooling
            # nn.AvgPool2d(kernel_size=3, stride=1, padding=1),

            # ### fc6 relu6 drop6
            # nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=12, dilation=12),
            # nn.ReLU(True),
            # nn.Dropout2d(0.5),

            # ### fc7 relu7 drop7 (kernel_size=1, padding=0)
            # nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            # nn.Dropout2d(0.5),

            # ### fc8
            # nn.Conv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        )
        # Initialize the avgpooling and classifier as standard VGG-16 so we can use the pretrained weights
        #TODO: replace this with the DeepLab avgpool and classifier, and built a custom state_dict loader for this model
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 1000)
        )

        ## DeepLab-style:
        # self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # self.classifier = nn.Linear(in_features=512, out_features=num_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d) and name == 'features.38':
                    nn.init.normal_(module.weight.data, mean=0, std=0.01)
                    nn.init.constant_(module.bias.data, 0.0)
