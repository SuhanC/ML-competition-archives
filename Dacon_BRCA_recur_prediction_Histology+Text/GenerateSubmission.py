import numpy as np
import pandas as pd

def getdata(npfile,testfile):
    og_prediction = np.load(npfile)
    testID = pd.read_csv(testfile).ID.tolist()
    predict_df_lst = [pd.DataFrame(df,index = testID) for df in og_prediction]
    return(predict_df_lst)

def most_common(lst):
    return max(set(lst), key=lst.count)

def get_most_common(predict_df_lst):
    df_lst=[]
    for df in predict_df_lst:
        df['N_category'] = df.idxmax(axis=1)
        df = df[['N_category']].reset_index()
        df_lst.append(df)
    common_lst=[]
    for i in list(range(len(df_lst[0]))):
        tmp=[]
        for d in df_lst:
            tmp.append(d.N_category.tolist()[i])
            most_freq = most_common(tmp)

        common_lst.append(most_freq)

    common_df = pd.DataFrame(common_lst,index = df.index.tolist()).reset_index()

    return(common_df)


def generate_submission_whole(common_df,outpath):
    common_df.columns = ['ID','N_category']
    common_df.ID = predict_df_lst[0].index.tolist()
    common_df.to_csv(outpath+'/submission_SHC.csv',index = False)
    return(0)

def generate_submission_perfold(df,suffix,outpath):
    df['N_category'] = df.idxmax(axis=1)
    df = df[['N_category']].reset_index()
    df.columns = ['ID','N_category']
    df.to_csv(outpath+'/submission.'+suffix+'.csv',index = False)
    return(0)



npfile = '/home/suhan/script/dancon_cancer/log_ml/lgbm.npy'
#testfile = '/home/suhan/data/Dacon_BRCA/test.csv'
testfile ='/home/suhan/data/Dacon_BRCA/20221124/test_clean.csv' 
submission_output = '/home/suhan/data/Dacon_BRCA/20221124/'
predict_df_lst = getdata(npfile,testfile)
common_df = get_most_common(predict_df_lst)
generate_submission_whole(common_df,submission_output)

predict_df_lst = getdata(npfile,testfile)
for i,df in enumerate(predict_df_lst):
    generate_submission_perfold(df,'Fold'+str(i),submission_output)
