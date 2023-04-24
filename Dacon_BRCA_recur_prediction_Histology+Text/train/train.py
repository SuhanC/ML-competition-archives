import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from model.plnet import LightModule
from datamodule.dm import CustomDataModule


def train(args, fold, train, test):
    dm= CustomDataModule(args, train, test)
    dm.set_fold_num(fold)
    dm.setup('fit')
    
    model= LightModule(args)
    
    earlystop= EarlyStopping(
        monitor= 'valid_loss',
        patience= args.hold,
        verbose= True,
        mode= 'min'
    )
    
    ckpt= ModelCheckpoint(
        dirpath= args.log_dir + f'/{args.model_name}_{fold}',
        monitor= 'valid_loss',
        verbose= True,
        mode= 'min'
    )
    
    lr_monitor= LearningRateMonitor(
        logging_interval= 'epoch',
        
    )
    
    train_cfg= dict(
        accelerator= args.is_gpu,
        devices= args.device,
        callbacks= [earlystop, ckpt, lr_monitor],
        precision= args.half,
        max_epochs= args.epochs
    )
    
    trainer= pl.Trainer(**train_cfg)
    trainer.fit(model, dm)