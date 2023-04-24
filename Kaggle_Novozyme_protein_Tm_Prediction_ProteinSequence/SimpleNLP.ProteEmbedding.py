import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch



inpath = '/home/suhan/data/kaggle/novozyme/'
train = pd.read_table(inpath+'/train.csv',sep=',').dropna()
test = pd.read_table(inpath+'/test.csv',sep=',')


train['prot_len'] = [len(p) for p in train.protein_sequence.tolist()]
test['prot_len'] = [len(p) for p in test.protein_sequence.tolist()]


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

test_df["data_source"] = test_df["data_source"].astype('category')
test_df["data_source_cat"] = test_df["data_source"].cat.codes

model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t6_8M_UR50D") # using relatively small model
model = model.cuda()


def GetDataForESM(df):
    result = [(df.seq_id.tolist()[i],df.protein_sequence.tolist()[i]) for i in range(len(df))]
    return(result)



def GetProteinRepresentation(data):
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    # with torch.no_grad():
    #     results = model(batch_tokens, repr_layers=[6], return_contacts=True)
    # token_representations = results["representations"][6]

    with torch.no_grad():
        results = model(batch_tokens.cuda(), repr_layers=[6])
    token_representations = results["representations"][6]

    # Generate per-sequence representations via averaging
    # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
    sequence_representations = []
    for i, tokens_len in enumerate(batch_lens):
        sequence_representations.append(token_representations[i, 1 : tokens_len - 1].mean(0))
    return(sequence_representations)


train_dict = GetDataForESM(train_df)
test_dict = GetDataForESM(test_df)


test_embedding = GetProteinRepresentation(test_dict)
train_embedding = GetProteinRepresentation(train_dict)


train_input = train_df.drop(['protein_sequence','data_source','seq_id'],axis=1).dropna()
test_input = test_df.drop(['protein_sequence','data_source','seq_id'],axis=1)



import optuna
import sklearn
import sklearn.ensemble
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# Define an objective function to be minimized.
def objective(trial):

    regressor_name = trial.suggest_categorical('classifier', ['RandomForest'])
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
    regressor_obj = sklearn.ensemble.RandomForestRegressor(max_depth=rf_max_depth)

    X_train, X_val, y_train, y_val = train_test_split(train_input.drop('tm',axis=1),train_input['tm'])
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.fit_transform(X_val)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = sklearn.metrics.mean_squared_error(y_val, y_pred)
    # from scipy import stats
    # print("Spearman correlation coef on unknown data is",str(stats.spearmanr(y_test, y2)[0]))

    return error  # An objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=5)  # Invoke optimization of the objective function.


clf = study.best_trial.params['classifier']
print(clf)
best_model = sklearn.ensemble.RandomForestRegressor(max_depth = study.best_trial.params['rf_max_depth'])
X_train, X_val, y_train, y_val = train_test_split(train_input.drop('tm',axis=1),train_input['tm'])
X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)
best_model.fit(X_train,y_train)

from scipy import stats
corr_value = stats.pearsonr(y_val,best_model.predict(X_val))[0]
print(corr_value)


test_input = test_df.drop(['protein_sequence','data_source','seq_id'],axis=1)
predicted_data = pd.DataFrame(best_model.predict(test_input))


predicted_data.index = test.seq_id
predicted_data.columns = ['tm']
predicted_data = predicted_data.reset_index()
predicted_data.to_csv('submission.csv',sep=',',index = False)