import sys
sys.path.append('')
import torch.optim as optim
from config import Config

def select_optim(args, params, lr, decay):

    if args.optim== 'sgd':
        return optim.SGD(params= params, lr= lr, weight_decay= decay)
    
    if args.optim== 'adam':
        return optim.Adam(params= params, lr= lr, weight_decay= decay)
    
    if args.optim== 'adamw':
        return optim.AdamW(params= params, lr= lr, weight_decay= decay)
    
    