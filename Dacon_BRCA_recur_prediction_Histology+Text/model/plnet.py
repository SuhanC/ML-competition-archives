import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
import pytorch_lightning as pl
# from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
# from utils.select_optim import select_optim
# from model.multi_modal import MultiModel
from utils.scheduler import CosineAnnealingWarmUpRestarts
from utils.focal_loss import FocalLoss

class LightModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args= args
        self.model= timm.create_model(model_name= args.model_name, num_classes= args.num_classes, pretrained= args.pretrained)
        self.f1= torchmetrics.F1Score(args.num_classes, 'weighted')
        self.loss= FocalLoss()
        self.value_list= list()
    
    def forward(self, img):

        return self.model(img)
    
    def configure_optimizers(self):

        optimizer= optim.SGD(params= self.parameters(), lr= self.args.lr, weight_decay= self.args.decay)
        scheduler= CosineAnnealingWarmUpRestarts(optimizer, self.args.t0, self.args.tmult, self.args.eta, self.args.up, self.args.gamma)
        return [optimizer], [scheduler]
    
    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, 'valid')
    
    def validation_epoch_end(self, output):
        
        avg_loss= torch.stack([x for x in output]).mean()
        self.log('avg_valid_loss', avg_loss)
        return avg_loss
    
    def test_step(self, batch, batch_idx):
        img = batch
        output= self(img)
        value= torch.softmax(output, 1).detach().cpu().numpy()
        self.value_list.extend(value)
    
    def _step(self, batch, step):

        img, label= batch
        output= self(img)
        loss= self.loss(output, label)
        f1= self.f1(output, label)
        
        self.log_dict(
            {
                f'{step}_loss': loss,
                f'{step}_f1': f1
            },
            prog_bar= True
        )
        
        return loss
        