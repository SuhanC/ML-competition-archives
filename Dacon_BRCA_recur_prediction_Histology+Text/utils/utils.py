import os
import numpy as np
import random
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
import pytorch_lightning as pl


def create_file(file):
    if os.path.exists(file):
        print('file exist')
    else:
        os.mkdir(file)
        
        
def seed_everything(seed):   
    print(f'Seed Everything')     
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    pl.seed_everything(seed)
    
def get_values(value):
    return value.values.reshape(-1, 1)

def scale_col(train, test, numeric_cols, ignore_cols):
    '''
    train : trainset dataframe
    test: testset dataframe
    numeric_cols: cols type int, float
    ignore_cols: clos type except int, float
    '''
    
    for col in train.columns:
        if col in ignore_cols:
            continue
        if col in numeric_cols:
            scaler= MinMaxScaler()
            train[col]= scaler.fit_transform(get_values(train[col]))
            test[col] = scaler.transform(get_values(test[col]))
        else:
            le = LabelEncoder()
            train[col] = le.fit_transform(get_values(train[col]))
            test[col] = le.transform(get_values(test[col]))
            
    return train, test