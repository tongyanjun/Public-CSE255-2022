from lib.KDTreeEncoding import *

import xgboost as xgb
from lib.XGBHelper import *
from lib.XGBoost_params import *
from lib.score_analysis import *

from lib.logger import logger

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from numpy import load
from glob import glob
import pandas as pd
import pickle as pkl

poverty_dir='/Users/yoavfreund/datasets/poverty_v1.1/'
image_dir=poverty_dir+'anon_images/'
#use the following line when running on datahub
#poverty_dir = '/dsmlp/datasets/cs255-sp22-a00-public/poverty/'
depth=8   #for KDTree

import pickle as pkl
pkl_file='data/DumpNoSplit.pk'
D=pkl.load(open(pkl_file,'rb'))
# ['styled_logs', 'tree', 'mean', 'std']
for k in D:
    globals()[k]=D[k]
scaling_mean=mean
scaling_std=std

bst_list=[x['bst'] for x in styled_logs[1]['log']]

### Iterate over test sets

folds=[{'in':'country_test_reduct.csv','out':'results_country.csv'},
      {'in':'random_test_reduct.csv','out':'results.csv'}]

#%%writefile generate_predictions.py
for fold_i in range(len(folds)):
    fold=folds[fold_i]

    #load table entries
    test_csv=f'../public_tables/{fold["in"]}'

    test=pd.read_csv(test_csv,index_col=0)
    test.index=test['filename']
    test.shape

    out=pd.DataFrame()
    out.index=test.index
    out['label']=0
    out['urban']=test['urban']

    ## Encode all data using encoding tree
    Enc_data=encoded_dataset(image_dir,out,tree,label_col='label')

    data=to_DMatrix(Enc_data.data)
    Preds=zeros([Enc_data.data.shape[0],len(bst_list)])
    for i in range(len(bst_list)):
        Preds[:,i]=bst_list[i].predict(data,output_margin=True)
    Preds=(Preds-scaling_mean)/scaling_std # apply overall score scaling

    _mean=np.mean(Preds,axis=1)
    _std=np.std(Preds,axis=1)

    pred_wo_abstention=(2*(_mean>0))-1
    pred_with_abstention=copy(pred_wo_abstention)
    pred_with_abstention[_std>abs(_mean)]=0

    out['label'] = pred_with_abstention
    out['label_accuracy'] = pred_wo_abstention


    outFile=f'data/{fold["out"]}'
    out.to_csv(outFile)
    print('\n\n'+'-'*60)
    print(outFile)
