import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
# from datamodule.dataset import CustomDataset
# from datamodule.dm import CustomDataModule

class Linear(nn.Module):
    def __init__(self, in_f, out_f, drop=False, p=None):
        super().__init__()
        self.linear= nn.Linear(in_f, out_f)
        self.bn= nn.BatchNorm1d(out_f)
        self.act= nn.SiLU()
        self.drop= drop
        
        if self.drop:
            self.out= nn.Dropout(p)
        
    def forward(self, x):
        x= self.linear(x)
        x= self.bn(x)
        x= self.act(x)
        x= self.drop(x)
        
        return x
    
class TabulrExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor= nn.Sequential(
            Linear(26, 32, False),
            Linear(32, 64, False),
            Linear(64, 128, False),
            Linear(128, 256, False),
            Linear(256, 512, False),
            Linear(512, 1024, True, 0.3)
    
        )
        
    def forward(self, x):
        return self.extractor(x)
    
class ImageExtractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args= args
        self.model= timm.create_model(model_name= self.args.model_name, pretrained= self.args.pretrained)
        self.embedding= nn.Linear(self.model.get_classifier().in_features, self.args.embedding_size)
    
    def forward(self, x):
        x= self.model(x)
        x= self.embedding(x)
        return x
    
class MultiModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args= args
        self.img_extractor= ImageExtractor(self.args)
        self.tabular_extractor= TabulrExtractor()
        self.classifier= nn.Sequential(
            nn.Linear(in_features= 1024, out_features= 2),
            nn.SiLU()
        )
        
    def forward(self, image, tabular):
        image_feature= self.img_extractor(image)
        tabular_feature= self.tabular_extractor(tabular)
        output= torch.cat([image_feature, tabular_feature], dim=-1)
        output= self.classifier(output)
        
        return output
    
    