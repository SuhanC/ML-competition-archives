class Config:
# deeplearning config
    root = '/home/suhan/data/Dacon_BRCA/'
    ### default
    seed= 42
    train_csv= root+'/20221124/train_clean.csv'
    test_csv= root+'/20221124/test_clean.csv'
    sub_csv= root+'sample_submission.csv'
    log_dir= root+'log'
    log_value= 'log_value'
    folds= 5
    target_col= 'N_category'
    image_col= 'img_path'
    
    ### for train
    epochs= 100
    img_size= 520
    batch_size= 8
    optim= 'sgd'
    lr= 4e-3
    decay= 5e-6
    embedding_size= 1024
    
    ### model
    model_name= 'tf_efficientnet_b5_ap'
    num_classes= 2
    pretrained= True
    
    ### scheduler
    t0= 5
    tmult= 2
    eta= 0.1
    up= 3
    gamma= 0.5
    
    ### trainer
    half= 32
    hold= 20
    is_gpu= 'gpu'
    device= -1 
    
# lgbm config
    lgbm_tr_csv= root+'/20221124/train_clean.csv'
    lgbm_ts_csv= root+'/20221124/test_clean.csv'
    lgbm_sub_csv= root+'/20221124/sample_submission.csv'
    log_lgbm= 'log_ml'
    lgbm_fold= 5 
    lgbm_trial= 1000 # OG 1000
    lgbm_period= 1000
    lgbm_stop= 1000
    
    
    

    
