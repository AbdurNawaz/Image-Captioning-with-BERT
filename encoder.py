import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    """
        We use ResNet 101 CNN as our encoder and do the following changes for our madel:

        1.Removing the last pooling and linear layers as we only need
          image encoding rather than image classification.

        2.We pass the output of this model onto an Adaptive pooling 
          layer to create a fixed size output vector that can be easily
          passsed to the decoder.
    """
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((14, 14))

    def forward(self, images):
        out = self.adaptive_pool(self.resnet(images))
        # batch_size, img size, imgs size, 2048
        out = out.permute(0, 2, 3, 1)
        return out

