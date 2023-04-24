from glob import glob
import numpy as np
from train.eval import test

def ensemble_predict(args):
    ensemble= list()
    ckpt_list= sorted(glob(f'{args.log_dir}/{args.model_name}*/*.ckpt'))
    fold_list= list(range(args.folds))
    
    for fold, ckpt in zip(fold_list, ckpt_list):
        result= test(ckpt, args, fold)
        result= np.array(result)
        ensemble.append(result)
    
    final= np.mean(ensemble, axis= 0)
    np.save(file= f'{args.log_value}/{args.model_name}', arr= final)
    print('=============================== npy file saved ===============================')
    