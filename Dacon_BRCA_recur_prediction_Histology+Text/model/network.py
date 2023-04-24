import torch
import torch.nn as nn
import timm

class Net(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net= timm.create_model(model_name= args.model_name, num_classes= args.num_classes, pretrained= args.pretrained)

    def forward(self, x):
        return self.net(x)