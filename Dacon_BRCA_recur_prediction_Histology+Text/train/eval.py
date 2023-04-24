import pytorch_lightning as pl
from datamodule.dm import CustomDataModule
from model.plnet import LightModule

def test(ckpt, args, fold):
    dm= CustomDataModule(args)
    dm.set_fold_num(fold)
    dm.setup('test')
    
    model= LightModule(args).load_from_checkpoint(checkpoint_path= ckpt, args= args)
    
    test_config= dict(
        accelerator= args.is_gpu,
        devices= args.device,
        precision= args.half
    )
    
    trainer= pl.Trainer(**test_config)
    trainer.test(model, dm)
    
    return model.value_list
    