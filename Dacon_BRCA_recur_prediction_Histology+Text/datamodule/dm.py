import os
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datamodule.dataset import CustomDataset, CustomMultiDataset
from utils.augment import train_augment, test_augment

class CustomDataModule(pl.LightningDataModule):
    def __init__(self, args, train, test):
        super().__init__()
        self.args= args
        self.train_df= train
        self.test_df= test
        # self.train_df= pd.read_csv(args.train_csv)
        # self.test_df= pd.read_csv(args.test_csv)
        
        if self.args.folds > 0:
            self.do_kfold()
        
    def set_fold_num(self, fold_num):
        self.fold_num= fold_num
    
    def do_kfold(self):
        skf= StratifiedKFold(n_splits= self.args.folds, shuffle= True, random_state= self.args.seed)
        for fold, (_, val) in enumerate(skf.split(X= self.train_df, y= self.train_df[self.args.target_col])):
            self.train_df.loc[val, 'kfold']= int(fold)
            
    def setup(self, stage):
        if stage== 'fit':
            print(f'Fold : {self.fold_num} Start')
            
            train_df= self.train_df[self.train_df['kfold'] != self.fold_num].reset_index(drop= True)
            valid_df= self.train_df[self.train_df['kfold'] == self.fold_num].reset_index(drop= True)
            
            self.train_ds= CustomDataset(train_df[self.args.image_col], train_df[self.args.target_col], train_augment(self.args.img_size))
            self.valid_ds= CustomDataset(valid_df[self.args.image_col], valid_df[self.args.target_col], train_augment(self.args.img_size))
        
        if stage== 'test':
            self.test_ds= CustomDataset(self.test_df[self.args.image_col], None, test_augment(self.args.img_size))
    
    def train_dataloader(self):
        return self._dataloader(self.train_ds, is_train= True)
    
    def val_dataloader(self):
        return self._dataloader(self.valid_ds, is_train= False)
    
    def test_dataloader(self):
        return self._dataloader(self.test_ds, is_train= False)
    
    def _dataloader(self, dataset, is_train= False):
        return DataLoader(dataset= dataset,
                          batch_size= self.args.batch_size,
                          num_workers= torch.cuda.device_count() * 4,
                          pin_memory= True,
                          shuffle= is_train
                          )