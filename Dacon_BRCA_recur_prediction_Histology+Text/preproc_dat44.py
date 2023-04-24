def annot_subtype(df):
    '''
    1 : Triple negative
    2 : Luminal A
    3 : Luminal B
    4 : HER2 Type
    5 : Other
    '''
    sub_lst=[]
    for i in range(len(df)):
        if (df.ER.tolist()[i]==0) & (df.PR.tolist()[i]==0) & (df.HER2.tolist()[i]==0):
            sub_lst.append('1')
        elif (df.ER.tolist()[i]==1) & (df.PR.tolist()[i]==1) & (df.HER2.tolist()[i]==0):
            sub_lst.append('2')
        elif (df.ER.tolist()[i]==1) & (df.PR.tolist()[i]==1) & (df.HER2.tolist()[i]==1):
            sub_lst.append('3')
        elif (df.ER.tolist()[i]==0) & (df.PR.tolist()[i]==0) & (df.HER2.tolist()[i]==1):
            sub_lst.append('4')
        else:
            sub_lst.append('5')
    df['Subtype'] = sub_lst

    return(df)


def fill_NaN(df):
    df['ER_Allred_score'].fillna(df['ER_Allred_score'].mean(),inplace=True) # impute mean
    df['PR_Allred_score'].fillna(df['PR_Allred_score'].median(),inplace = True) # impute median
    df["암의 장경"] = df["암의 장경"].fillna(0.1)
    df['T_category'] = df['T_category'].fillna(1) # Imputate T category with 1 bc at least cancer's stage 1

    ################# Temporal Imputation ###########
    df["KI-67_LI_percent"] = df["KI-67_LI_percent"].fillna(df["KI-67_LI_percent"].mean())
    df['HER2_IHC'].fillna(4,inplace = True)
    df['HER2_SISH'].fillna(4,inplace = True)
    df['HER2'].fillna(4,inplace = True)
    df['ER'].fillna(4,inplace = True)
    df['PR'].fillna(4,inplace = True)

    df['DCIS_or_LCIS_type'].fillna(4,inplace = True)
    df['BRCA_mutation'].fillna(4,inplace = True)
    # From EJJ
    for i in ['NG', 'HG','HG_score_1','HG_score_2', 'HG_score_3']:
        df[i] = df[i].fillna(1)
    return(df)

def remove_cols(df,rm_cols):
    df_out = df.loc[:,~df.columns.isin(rm_cols)]
    return(df_out)

def scale_columns(df,scale_cols):

    scaler = StandardScaler()

    df_noscale = df.loc[:,~df.columns.isin(scale_cols)]
    df_scale = df[scale_cols]

    scaler.fit(df_scale)
    df_scale = scaler.transform(df_scale)
    df_scale = pd.DataFrame(df_scale,columns=scale_cols)

    df_out = pd.concat([df_noscale,df_scale],axis=1)

    return(df_out)


def yield_output(train,test,outpath):
    os.makedirs(outpath+'/'+year+month+date,exist_ok = True)
    outpath = outpath+'/'+year+month+date+'/'
    train.to_csv(outpath+'train_clean.csv',sep=',',index = False)
    test.to_csv(outpath+'test_clean.csv',sep=',',index = False)
    return(0)


scale_cols = ['암의 장경','ER_Allred_score',
              'PR_Allred_score','HG_score_1',
              'HG_score_2', 'HG_score_3',
              'KI-67_LI_percent']
rm_cols = ['HER2_IHC','HER2_SISH','HER2_SISH_ratio','BRCA_mutation','DCIS_or_LCIS_type','KI-67_LI_percent']
rm_cols = ['HER2_SISH_ratio']

import pandas as pd
import numpy as np
import datetime
import os
from sklearn.preprocessing import StandardScaler

year = str(datetime.datetime.today().year)
month = str(datetime.datetime.today().month)
date = str(datetime.datetime.today().day)


tmetadata = pd.read_excel('/home/suhan/data/Dacon_BRCA/clinical_info.xlsx')
train = pd.read_table('/home/suhan/data/Dacon_BRCA/train_pathadded.csv',sep=',')
test = pd.read_table('/home/suhan/data/Dacon_BRCA/test_pathadded.csv',sep=',')


train_ann = annot_subtype(train) ; test_ann = annot_subtype(test)
train_ann = fill_NaN(train_ann) ; test_ann = fill_NaN(test_ann)
train_ann = remove_cols(train_ann,rm_cols) ; test_ann = remove_cols(test_ann,rm_cols)
train_ann = scale_columns(train_ann,scale_cols) ; test_ann = scale_columns(test_ann,scale_cols)
yield_output(train_ann,test_ann,'/home/suhan/data/Dacon_BRCA/')

