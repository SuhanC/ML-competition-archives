import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import optuna
import sklearn
import sklearn.ensemble
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

DATADIR = '/home/suhan/data/Dacon_BRCA/train_segmented/'
DATADIR_OG = '/home/suhan/data/Dacon_BRCA/train_imgs/'
DATADIR_MSK = '/home/suhan/data/Dacon_BRCA/train_masks/'

masked_files = [DATADIR_MSK+f for f in os.listdir(DATADIR_MSK)]

IMG_SIZE=[1500,1000]
clinic_dat = pd.read_table('/home/suhan/data/Dacon_BRCA/20221124/train_clean.csv',sep=',')

def imCrop(x):
    height,width,depth = x.shape
    return [x[height , :width//2] , x[height, width//2:]]

def create_training_data():
    training_data=[]
    for c in range(len(clinic_dat)):
        img_id = clinic_dat.ID.tolist()[c]
        img_array_og = cv2.imread(DATADIR_OG+img_id+'.png')

        if not masked_files.count(DATADIR_MSK+img_id+'.png'):
            print('Using generated masking data : '+img_id)
            img_array=cv2.imread(DATADIR+img_id+'.png')
        else : 
            print('Using original masking data : '+img_id)
            img_array=cv2.imread(DATADIR_MSK+img_id+'.png')
        
        new_array=cv2.resize(img_array,(IMG_SIZE[0],IMG_SIZE[1]))
        new_array_og=cv2.resize(img_array_og,(IMG_SIZE[0],IMG_SIZE[1]))

        new_array = new_array[:,IMG_SIZE[1]//2]
        new_array_og = new_array_og[:,IMG_SIZE[1]//2]

        # https://bkshin.tistory.com/entry/OpenCV-9-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%97%B0%EC%82%B0 # Image synthesis
        alpha = 0.8 # Hard-coded to set alpha blending parameter
        blended = new_array * alpha + new_array * (1-alpha)
        synth_img = blended.astype(np.uint8)

        # training_data.append([new_array,clinic_dat.N_category.tolist()[c]])
        training_data.append([synth_img,clinic_dat.N_category.tolist()[c]])
    return(training_data)


def preprocess_test_data(testpath,testpath_seg):
    test_data=[]
    filenames=[]
    test_names = os.listdir(testpath)
    for f in test_names:
        img_array_og = cv2.imread(testpath+f)
        img_array=cv2.imread(testpath_seg+f)
        
        new_array=cv2.resize(img_array,(IMG_SIZE[0],IMG_SIZE[1]))
        new_array_og=cv2.resize(img_array_og,(IMG_SIZE[0],IMG_SIZE[1]))

        new_array = new_array[:,IMG_SIZE[1]//2]
        new_array_og = new_array_og[:,IMG_SIZE[1]//2]

        # https://bkshin.tistory.com/entry/OpenCV-9-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EC%97%B0%EC%82%B0 # Image synthesis
        alpha = 0.8 # Hard-coded to set alpha blending parameter
        blended = new_array * alpha + new_array * (1-alpha)
        synth_img = blended.astype(np.uint8)

        test_data.append(synth_img)
        filenames.append(f)
    return(test_data,filenames)


TESTDIR = '/home/suhan/data/Dacon_BRCA/test_imgs/'
TESTDIR_SEG = '/home/suhan/data/Dacon_BRCA/test_segmented/'
test_data, testfiles = preprocess_test_data(TESTDIR,TESTDIR_SEG)
test_data= np.array(test_data).reshape(len(test_data),-1)

training_data = create_training_data()


print('Train data set done')

X=[]
y=[]
for categories, label in training_data:
    X.append(categories)
    y.append(label)
X= np.array(X).reshape(len(X),-1)
y=np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X,y)

def objective(trial):

    regressor_name = trial.suggest_categorical('classifier', ['RandomForest'])
    rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
    regressor_obj = sklearn.ensemble.RandomForestClassifier(max_depth=rf_max_depth,n_jobs=15)

    print('Training Start')
    from sklearn.model_selection import train_test_split
    X=[]
    y=[]
    for categories, label in training_data:
        X.append(categories)
        y.append(label)
    X= np.array(X).reshape(len(X),-1)
    y=np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X,y)
  
    regressor_obj.fit(X_train, y_train)
    y2 = regressor_obj.predict(X_test)

    error = sklearn.metrics.accuracy_score(y_test, y2)
    return error  # An objective value linked with the Trial object.

study = optuna.create_study()  # Create a new study.
study.optimize(objective, n_trials=10)  # Invoke optimization of the objective function.

clf = study.best_trial.params['classifier']
print(clf)
best_model = sklearn.ensemble.RandomForestClassifier(max_depth = study.best_trial.params['rf_max_depth'])

best_model.fit(X_train,y_train)
y2 = best_model.predict(X_test)
test_result = best_model.predict(test_data)

print("Accuracy on unknown data is\n",accuracy_score(y_test,y2))
print("Accuracy on unknown data is\n",classification_report(y_test,y2))

result = pd.DataFrame({'original' : y_test,'predicted' : y2})
result.to_csv('./result.tsv',sep='\t',index = False)

submission = pd.DataFrame({'ID' : [t.split('.')[0] for t in testfiles],'N_category' : test_result})
submission.to_csv('./Submission.tsv',sep=',',index = False)
