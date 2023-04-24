import pandas as pd
from utils.utils import create_file, seed_everything, scale_col
from train.train import train
from train.ensemble import ensemble_predict
from config import Config

if __name__ == '__main__':

    seed_everything(Config.seed)
    create_file(Config.log_dir)
    create_file(Config.log_value)
    
    train_df= pd.read_csv(Config.train_csv)
    test_df= pd.read_csv(Config.test_csv)
    numeric_cols = ['나이', '암의 장경', 'ER_Allred_score', 'PR_Allred_score', 'KI-67_LI_percent', 'HER2_SISH_ratio']
    ignore_cols = ['ID', 'img_path', 'mask_path', '수술연월일', 'N_category']
    
    train_df,test_df = scale_col(train_df, test_df, numeric_cols, ignore_cols)
 
    for fold in range(Config.folds):
        train(Config, fold, train_df, test_df)
        
    ensemble_predict(Config)

    # x= np.load('log_value/tf_efficientnet_b4_ap.npy')
    # x= np.argmax(x, 1)
    # fin= pd.read_csv(Config.sub_csv)
    # fin[Config.target_col]= x
    # fin.to_csv('b4.csv', index=False)