import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pylab import *
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split
from numpy.random import choice
from lib.XGBoost_params import *



class DataLoader:
    def __init__(self,data,TrainTest=0.5,TestVal=0.001):
        print(f'data shape={data.shape}')

        X = data[:, :-1]
        y = np.array(data[:, -1], dtype=int)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TrainTest)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=TestVal)

        print(f'train_size={X_train.shape[0]} validation_size={X_val.shape[0]}, test_size={X_test.shape[0]}')
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, y_val)
        dtest = xgb.DMatrix(X_test, label=y_test)
        self.S={'data':data,
           'X':X,'y':y,
           'X_train':X_train, 'y_train':y_train, 
           'X_test':X_test, 'y_test':y_test, 
           'X_val':X_val,   'y_val':y_val,
           'dtrain':dtrain, 'dval':dval, 'dtest':dtest      
          }
    def get(self,L):
        answer=[]
        for l in L:
            assert l in self.S, f'DataLoader.get: no item named {l} in {self.S.keys()}'
            answer.append(self.S[l])
        return answer

def plot_log(Log):
    figure(figsize=(12,5))
    i=1
    for loss in ['error','logloss']:
        subplot(1,2,i); i+=1
        for dataset in ['eval','train']:
            _label='%s-%s'%(dataset,loss)
            plot(Log[dataset][loss],label=_label)
        _argmin=argmin(Log['eval'][loss])
        _min=Log['eval'][loss][_argmin]
        _title=f"min of eval-{loss}={_min} at {_argmin}"
        title(_title)
        legend()
        grid()

def test_xgboost(data,param,depth=4,num_round=100):
    D=DataLoader(data)
    dtrain,dval,dtest = D.get(['dtrain','dval','dtest'])
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    param['max_depth']= depth   # depth of tree
    evals_result={}

    bst=xgb.train(param_D2L(param), dtrain, num_round, evallist,\
                verbose_eval=False, evals_result=evals_result)
    plot_log(evals_result)
    return bst
       
def plot_roc(y_test,scores):
    plt.figure(figsize=(8, 6))

    for y_scores,label in scores:
        tpr,fpr, thresholds = roc_curve(y_test, y_scores)
        plt.plot(fpr, tpr, label=label)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title("ROC Curves")
    plt.grid()
    plt.legend()
    plt.show()


def bootstrap_sample(X,y):
    assert X.shape[0]==y.shape[0]
    l=X.shape[0]

    C=choice(array(range(l)),l,replace=True)
    Xresamp=X[C]
    yresamp=y[C]
    return Xresamp,yresamp

    
def plot_margins(X_train,y_train,X_test,y_test,param,ensemble_size=20,TrainingRounds=[10,200],\
                 labels=['Rich','Poor'],_percentile=0.1):
    plt.figure(figsize=(8, 6))
    models={iters:[] for iters in TrainingRounds}
    error_D={}

    dtest = xgb.DMatrix(X_test, label=y_test)
    for i in range(ensemble_size):  #iterate over randomized training of the classifier

        legends=[]
        for num_round in TrainingRounds:  # Number of training iterations
            X_train_bootstrap,y_train_bootstrap=bootstrap_sample(X_train,y_train)
            dtrain = xgb.DMatrix(X_train_bootstrap, label=y_train_bootstrap)
            evallist = [(dtrain, 'train'), (dtest, 'eval')]
            bst = xgb.train(param_D2L(param), dtrain, num_round, evallist, verbose_eval=False)
            
            models[num_round].append(bst)
            y_pred = bst.predict(dtest,output_margin=True)
            thresholds = sorted(np.unique(np.round(y_pred, 2)))
            #thresholds = thresholds/(np.percentile(thresholds,100-_percentile)-np.percentile(thresholds,_percentile))
            
            error_pos, error_neg = get_error_values(y_pred, y_test, thresholds)
            legends += [f'{labels[0]}{num_round}', f'{labels[1]}{num_round}']
            _style=['y','g'] if num_round==10 else ['b', 'r']

            thresholds=thresholds/(np.max(thresholds)-np.min(thresholds))
            get_margin_plot(error_pos, error_neg, thresholds, legends = legends, style=_style)
            error_D[(i,num_round)]={'thresholds':thresholds,
                'error_pos':error_pos,
                'error_neg':error_neg}
            
        plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        plt.grid(which='minor', linestyle=':', linewidth='0.5', color='gray')

    plt.title('train size=%4.3f, test size=%4.3f'%(X_train.shape[0],X_test.shape[0]))
    plt.show()
    return error_D

    
def visualize_features(bst, features_map = None):
    xgb.plot_importance(bst)
    plt.show()
    if features_map:
        print("Feature Mapping:")
        for x in features_map:
            print(x, "\t: ", features_map[x])

def get_error_values(y_pred, y_test, thresholds):
    accuracy_1 = []
    accuracy_0 = []
    for thresh in thresholds:
        y_test_i = y_test[y_test == 1]
        y_pred_i = y_pred[y_test == 1]
        correct = np.sum(y_pred_i > thresh)
        accuracy_1.append(1.0 * correct / len(y_test_i))

        y_test_i = y_test[y_test == 0]
        y_pred_i = y_pred[y_test == 0]
        correct = np.sum(y_pred_i <= thresh)
        accuracy_0.append(1.0 * correct / len(y_test_i))
    
    error_1 = list(1 - np.array(accuracy_1))
    error_0 = list(1 - np.array(accuracy_0))
    return error_1, error_0

def get_margin_plot(error_1, error_0, thresholds, legends = None, title=None, style=['b', 'r']):
    #thresholds=thresholds/(np.max(thresholds)-np.min(thresholds))
    plt.plot(thresholds, error_1, style[0])
    plt.plot(thresholds, error_0, style[1])
    if legends:
        plt.legend(legends)
    plt.xlabel('Margin Score')
    plt.ylabel('Error %')
    if title:
        plt.title(title)

def statistics(y_pred, y_test, thr_lower, thr_upper):
    true_index = y_pred > thr_upper
    y_true =  np.sum(y_test[true_index] == 1)
    
    false_index = y_pred < thr_lower
    y_false =  np.sum(y_test[false_index] == 0)
    
    abstain = 1 - np.sum((y_pred < thr_lower) | (y_pred > thr_upper))/len(y_test)
    
    return (y_true+ y_false)/(len(y_test[true_index])+len(y_test[false_index])) , y_true/len(y_test[true_index]), y_false/len(y_test[false_index]), abstain


