import sys
sys.path.append('')
import pandas as pd
import numpy as np
import lightgbm
from lightgbm import log_evaluation, early_stopping

import optuna
from optuna import Trial
from optuna.samplers import TPESampler

from config import Config
from utils.utils import create_file
from utils.ml_util import preprocess_csv, create_skf
from model.lgbm_infer import lgbm_infer

train= preprocess_csv(Config.train_csv, 'train')
train= create_skf(args= Config, df= train)
test= preprocess_csv(Config.test_csv, 'test')

target= train[Config.target_col]
features= train.drop([Config.target_col,'kfold'], axis=1)


def objective(trial: Trial):
    score_list= list()

    params= {
    'objective': 'binary',
    'random_state': Config.seed,
    'boosting_type' : 'gbdt',
    "n_estimators" : trial.suggest_int('n_estimators', 100, 10000),
    'max_depth':trial.suggest_int('max_depth', 4, 1024),
    'seed': Config.seed,
    'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0),
    'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0),
    'num_leaves': trial.suggest_int('num_leaves', 4, 1024),
    'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 4, 1024),
    'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
    'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 1.0),
    'bagging_freq': trial.suggest_int('bagging_freq', 4, 1024),
    'min_child_samples': trial.suggest_int('min_child_samples', 4, 1024),
    'learning_rate': trial.suggest_float('learning_rate', 5e-4, 1e-1),
    'n_jobs': 25,
    }
    
    callbacks= [log_evaluation(period=Config.lgbm_period), early_stopping(stopping_rounds=Config.lgbm_stop)]

    for k in range(Config.lgbm_fold):

        print("#################################")
        print(f'######### {k + 1} FOLD START ##########')
        print("#################################")

        train_idx= train['kfold'] != k
        valid_idx= train['kfold'] == k

        X_train= features[train_idx].values
        y_train= target[train_idx].values

        X_valid= features[valid_idx].values
        y_valid= target[valid_idx].values


        model= lightgbm.LGBMClassifier(**params)
        model.fit(X_train, y_train, 
        eval_set= [(X_valid, y_valid)], 
        eval_metric='binary_logloss', 
        callbacks= callbacks)

        best_loss= model.best_score_['valid_0']['binary_logloss']
        score_list.append(best_loss)
    
    loss_mean= np.mean(score_list)

    return loss_mean



sampler= TPESampler(seed=Config.seed)
study= optuna.create_study(direction='minimize', sampler= sampler)
study.optimize(objective, n_trials=Config.lgbm_trial)
print('##############################################')
print("Best Score: ", study.best_value)
print('Best trial: ', study.best_trial.params)
best_trial= study.best_trial.params
print('##############################################')

create_file(Config.log_lgbm)
lgbm_infer(Config, train, features, target, test, best_trial,True)
