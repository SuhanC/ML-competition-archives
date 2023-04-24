import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


inpath = '/home/suhan/data/kaggle/novozyme/'
train = pd.read_table(inpath+'train.csv',sep=',').dropna()
test = pd.read_table(inpath+'test.csv',sep=',')


train['prot_len'] = [len(p) for p in train.protein_sequence.tolist()]


aatable = {
            'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T',
            'ACG':'T', 'ACT':'T', 'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
            'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R', 'CTA':'L', 'CTC':'L',
            'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
            'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R',
            'CGG':'R', 'CGT':'R', 'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
            'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 'GAC':'D', 'GAT':'D',
            'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
            'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F',
            'TTA':'L', 'TTG':'L', 'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
            'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
            }



aa = list(set(aatable.values()))


def aa_bow(sequence,aminolist):
    bow_in=[]
    for a in aminolist:
        bow_in.append(sequence.count(a))
    return(bow_in)



def get_bow(df):
    bow_lst=[]
    for i in range(len(df)):
        bow_tmp = aa_bow(df.protein_sequence.tolist()[i],aa)
        bow_lst.append(bow_tmp)
    bow_df = pd.DataFrame(bow_lst,columns = aa)
    return(bow_df)




train_df = pd.concat([get_bow(train),train],axis=1)
test_df = pd.concat([get_bow(test),test],axis=1)



train_df["data_source"] = train_df["data_source"].astype('category')
train_df["data_source_cat"] = train_df["data_source"].cat.codes

train_input = train_df.drop(['protein_sequence','data_source','seq_id'],axis=1).dropna()
test_input = test_df.drop(['protein_sequence','data_source','seq_id'],axis=1)





from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



import optuna
import sklearn
import sklearn.ensemble
from sklearn.model_selection import train_test_split


# # Define an objective function to be minimized.
# def objective(trial):

#     regressor_name = trial.suggest_categorical('classifier', ['SVR', 'RandomForest'])
#     if regressor_name == 'SVR':
#         svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
#         regressor_obj = sklearn.svm.SVR(C=svr_c)
#     else:
#         rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
#         regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

#     X_train, X_val, y_train, y_val = train_test_split(train_input.drop('tm',axis=1),train_input['tm'])
#     X_train = scaler.fit_transform(X_train)
#     X_val = scaler.fit_transform(X_val)

#     regressor_obj.fit(X_train, y_train)
#     y_pred = regressor_obj.predict(X_val)

#     error = sklearn.metrics.mean_squared_error(y_val, y_pred)
#     corr_value = stats.spearmanr(y_val, y_pred)[0]
#     print("Spearman correlation coef on unknown data is",str(stats.spearmanr(y_val, y_pred)[0]))

#     return error  # An objective value linked with the Trial object.

# study = optuna.create_study()  # Create a new study.
# study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.


from optuna import Trial, visualization
from optuna.samplers import TPESampler

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

def objectiveXGB(trial: Trial, X_train, y_train, y_test):
    param = {
        "n_estimators" : trial.suggest_int('n_estimators', 500, 4000),
        'max_depth':trial.suggest_int('max_depth', 8, 16),
        'min_child_weight':trial.suggest_int('min_child_weight', 1, 300),
        'gamma':trial.suggest_int('gamma', 1, 3),
        'learning_rate': 0.01,
        'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.5, 1, 0.1),
        'nthread' : 20,
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor',
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_categorical('subsample', [0.6,0.7,0.8,1.0] ),
        'random_state': 42
    }
    ## XGBRegressor를 예시로 사용하였습니다.
    model = XGBRegressor(**param)
    

    xgb_model = model.fit(X_train, y_train, verbose=False)
    y_pred = xgb_model.predict(X_val)

    ## RMSE으로 Loss 계산
    score = mean_squared_error(y_pred, y_test, squared=False)

    return score

# direction : score 값을 최대 또는 최소로 하는 방향으로 지정 
study = optuna.create_study(direction='minimize',sampler=TPESampler())
# n_trials : 시도 횟수 (미 입력시 Key interrupt가 있을 때까지 무한 반복)

X_train, X_val, y_train, y_val = train_test_split(train_input.drop('tm',axis=1),train_input['tm'])
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
study.optimize(lambda trial : objectiveXGB(trial, X_train,  y_train, y_val), n_trials=5)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value,study.best_trial.params))
best_model = XGBRegressor(**study.best_trial.params)
best_model.fit(X_train,y_train)
print("Spearman correlation coef on best XGBoost model is",str(stats.spearmanr(y_val, best_model.predict(X_val))[0]))
