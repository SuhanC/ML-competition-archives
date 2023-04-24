import cv2
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, img_list, label_list=None, augment=None):
        self.img_list= img_list
        self.label_list= label_list
        self.augment= augment
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img= self.img_list[idx]
        img= cv2.imread(img)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.augment is not None:
            img= self.augment(image= img)['image']
            
        # training
        if self.label_list is not None:
            label= self.label_list[idx]
            return img, torch.tensor(label)
        
        # test
        else:
            return img
        


class CustomMultiDataset(Dataset):
    def __init__(self, df, args, label_list=None, augment=None):
        self.df= df
        self.args= args
        self.label_list= label_list
        self.augment= augment
        
    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, idx):
        img= self.df[self.args.img_path][idx]
        img= cv2.imread(img)
        img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.augment is not None:
            img= self.augment(image= img)['image']
            
        # training
        if self.label_list is not None:
            label= self.label_list[idx]
            tabular= torch.tensor(self.df.drop(columns= None).iloc[idx])
            return img, torch.tensor(label), tabular
        
        # test
        else:
            tabular= torch.Tensor(self.df.drop(columns= None).iloc[idx])
            return img, tabular
        