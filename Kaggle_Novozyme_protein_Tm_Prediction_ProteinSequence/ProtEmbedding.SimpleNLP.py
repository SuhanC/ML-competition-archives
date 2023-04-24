import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings("ignore")


inpath = '/home/suhan/data/kaggle/novozyme/'
train_input = pd.read_table(inpath+'train_clean.csv',sep=',').dropna()
test_input = pd.read_table(inpath+'test_clean.csv',sep=',')
test = pd.read_table(inpath+'test.csv',sep=',')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()



import optuna
import sklearn
import sklearn.ensemble
from sklearn.model_selection import train_test_split

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

X_test = scaler.fit_transform(test_input)

predicted_data = pd.DataFrame(best_model.predict(X_test))

sns.histplot(predicted_data)

predicted_data.index = test.seq_id
predicted_data.columns = ['tm']
predicted_data = predicted_data.reset_index()
predicted_data.to_csv('submission.csv',sep=',',index = False)