import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

def train_augment(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT,p=0.3),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(),
        A.Blur(),
        A.Normalize(),
        ToTensorV2()
    ])
    
    
def test_augment(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(),
        A.Normalize(),
        ToTensorV2()
    ])