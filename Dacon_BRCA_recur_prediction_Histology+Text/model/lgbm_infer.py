import sys
sys.path.append('')
import numpy as np
import lightgbm
from lightgbm import log_evaluation, early_stopping

from config import Config
from utils.utils import create_file


def lgbm_infer(args, train, features, target, test, best, is_save):
    pred_proba= list()
    for k in range(args.folds):

        print("#################################")
        print(f'######### {k + 1} FOLD START ##########')
        print("#################################")

        train_idx= train['kfold'] != k
        valid_idx= train['kfold'] == k

        X_train= features[train_idx].values
        y_train= target[train_idx].values

        X_valid= features[valid_idx].values
        y_valid= target[valid_idx].values


        model= lightgbm.LGBMClassifier(**best)
        callbacks= [log_evaluation(period=args.lgbm_period), early_stopping(stopping_rounds=args.lgbm_stop)]

        model.fit(X_train, y_train, 
        eval_set= [(X_valid, y_valid)], 
        eval_metric='binary_logloss', 
        callbacks= callbacks)


        pred_proba.append(model.predict_proba(test))

    if is_save:
        print('Value Saved')
        np.save(args.log_lgbm + f'/lgbm',np.array(pred_proba))