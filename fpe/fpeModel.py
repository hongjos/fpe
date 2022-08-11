########################################
##
import pandas as pd
import numpy as np
import traceback
import os.path
import os
import pickle
import time
import datetime
from datetime import datetime
from timeit import default_timer as timer
import psutil
from scipy.stats import pearsonr

from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier
from sklearn import linear_model
from xgboost import XGBRegressor
from xgboost.sklearn import XGBClassifier
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from random import choices
import random

from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import datasets
from datasets import load_dataset, Dataset, DatasetDict
from tqdm import tqdm

from fpe import *
from fpeDataSource import *

########################################
# CONSTANT and setups
USE_ALL_TRAINING_SAMPLE = True

USE_InitModel = False

MODEL_NAME_DNN = 'DNN'
MODEL_NAME_LGB = 'LGB'
MODEL_NAME_XGB = 'XGBoost'

########################################
USETRAIN_SET_IN_PERCENT = 80
USETRAIN_FEATURE_CORR = 0.01

########################################
# base model
class ModelBase(object):
    def __init__(self, dataSource=None, name='model', opt=None):
        # dataSource should always be provided even it is an empty one=
        self.logger = logging.getLogger('fpe.model')
        if (dataSource is None):
            dataSource = DataSource();
        self.name = name ;
        self.opt = opt;
        self.ds = dataSource ;

        self.model = {}
        self.modelType = {}
        self.modelWeight = {} ;
        self.modelOpt = {}

    ################
    # to be defined in each subclass
    def setDataSource(self, dataSource):
        self.ds = dataSource;
    
    ################
    # to be defined in each subclass
    def buildModel(self, mtype, X, Y, opt=None):
        raise NotImplementedError("buildModel error message")       

    ################
    # to be defined in each subclass
    def train(self, dataSource=None, opt=None):
        raise NotImplementedError("buildModel error message")       

    ################
    def getModelName(self, name=None ):
        if name is None:
            name='noname'
        cmn = f'model_{name}.p'
        return cmn

    ################
    def getModelOpt(self, name=None ):
        if name is None:
            name='noname'
        cmn = f'model_{name}_opt.p'
        return cmn

    ################
    def writeModel(self, name=None):
        for key in self.model:
            cmn = self.getModelName(name=name)
            self.logger.info(f'writeModel : {cmn}')
            with open(cmn, 'wb') as f:
                # Pickle the 'data' dictionary using the highest protocol available.
                pickle.dump(self.model[key], f, pickle.HIGHEST_PROTOCOL) ;

    ################
    def readModel(self, name=None):
        cmn = self.getModelName(name=name)
        if (os.path.exists(cmn)):
            self.logger.info(f'readModel : {cmn}')
            with open(cmn, 'rb') as f:
                self.model[name] = pickle.load(f)

    ################
    def writeModelOpt(self, name=None):
        for key in self.model:
            cmn = self.getModelOpt(name=name)
            self.logger.info(f'writeModel opt : {cmn}')
            with open(cmn, 'wb') as f:
                pickle.dump(self.modelOpt[key], f, pickle.HIGHEST_PROTOCOL) ;

    ################
    def readModelOpt(self, name=None):
        cmn = self.getModelOpt(name=name)
        if (os.path.exists(cmn)):
            self.logger.info(f'readModel opt : {cmn}')
            with open(cmn, 'rb') as f:
                self.modelOpt[name] = pickle.load(f)

    ################
    def calcPearson(self, preds, target):
        #
        target = np.nan_to_num(target)
        return pearsonr(target, np.nan_to_num(preds))[0]


    ################
    def getFeatureDS(self, start=None, end=None ):
        tk_ds, target = self.ds.getFeature( start=start, end=end )
        return tk_ds, target

## end class ModelBase
####################################

####################################
class ModelGeneric(ModelBase):
    def __init__(self, dataSource=None, name='model', opt=None):
        super(ModelGeneric, self).__init__(dataSource=dataSource, name=name, opt=opt)
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        
    ################
    def buildModel(self, mtype=MODEL_NAME_LGB, X=None, Y=None, opt=None):
        if (mtype is None) or (mtype == MODEL_NAME_LGB) :
            if (opt is None):
                opt = {};
                opt['boosting_type'] = 'dart'
                opt['objective'] = 'multiclass'
                opt['max_bin'] = 256
                opt['num_leaves'] = 253
                opt['learning_rate'] = 0.006
                opt['n_estimators'] = 2500
                opt['min_child_samples'] = 100
                opt['max_depth'] = 9
            model = LGBMClassifier(**opt)
        elif (mtype == MODEL_NAME_XGB):
            if (opt is None):
                opt = {};
                opt['booster'] = 'dart' #'dart' #gbtree gblinear
                opt['n_estimators'] = 2000
                opt['max_depth'] = 15 
                opt['learning_rate'] = 0.05
                opt['min_child_weight'] = 50
                opt['gamma'] = 100.0 
                opt['lambda'] = 100
                #opt['top_k'] = 100
                opt['subsample'] = 0.9
                opt['colsample_bytree'] = 0.7
                #opt['missing'] =-1, 
                #opt['eval_metric']='rmse'
                # USE CPU
                #nthread=4,
                #tree_method='hist' 
                # USE GPU
                #opt['tree_method']='gpu_hist' 
                opt['verbosity'] = 2
            model = XGBClassifier(**opt) 
        else:
            if (opt is None):
                opt = {}
            model = LGBMClassifier(**opt)
        self.logger.info(f'Model = {mtype} opt = {str(opt)}')
        return model

    def getFeatureLM(self, xt, yt, hasTarget=True):
        model = AutoModelForSequenceClassification.from_pretrained(USE_MODEL_PATH_SAVE, num_labels=3)
        numLine = xt['input_ids'].shape[0]
        #numLine = 300
        if hasTarget:
            y = yt
            y = yt[:numLine]
        else:
            y = None
        bsize = 10
        x = np.zeros((0, 768))
        #for i in tqdm(range(0, len(yt), bsize)):
        for i in tqdm(range(0, numLine, bsize)):
            k = i + bsize
            t = {}
            t['input_ids'], t['attention_mask'], t['token_type_ids'] = xt['input_ids'][i:k], xt['attention_mask'][i:k], xt['token_type_ids'][i:k]
            output = model(**t, output_hidden_states=True)
            x = np.concatenate((x, output.hidden_states[6][:,0,:].detach().numpy()))
        return x, y

    ################
    def fitModel(self, name=MODEL_NAME_LGB, mtype=None, X=None, Y=None, init_model=None):
        if name is None:
            name = self.name;
        self.modelType[name] = mtype ;
        if (mtype is None) or (mtype == MODEL_NAME_LGB):
            self.model[name].fit(X, Y, eval_metric='logloss', init_model=init_model )
        elif (mtype == MODEL_NAME_XGB):
            self.model[name].fit(X, Y)
        else:
            self.model[name].fit(X, Y, eval_metric='logloss', init_model=init_model)

    ################
    def predictOneModel(self, name, x):
        try:
            y = self.model[name].predict_proba( x )
            return y
        except:
            self.logger.warning(f'predict error' ) ;
            traceback.print_exc()
            return 0

    def predict(self, x):
        yp = np.zeros((x.shape[0], 3))
        w = 0
        for name, model in self.model.items():
            y = self.predictOneModel(name, x)
            yp = yp + y ;
            if (self.modelWeight):
                w = w + self.modelWeight[name];
            else:
                w = w + 1;
        if (w != 0):
            yp = yp / w ;
        return yp ;

    ################
    # train on all data(split train/test)
    def train(self, xtrain=None, ytrain=None, xtest=None, ytest=None):
        try:
            from sklearn.metrics import log_loss
            if xtrain is None:
                self.ds.printMemoryUssage('train')
                import torch.nn.functional as F
                def score(preds): return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}
        
                trn_idxs, val_idxs = self.ds.getTrainTestSplit(0.2)
                ds, target = self.ds.getFeature()
                xt = {}
                xs = {}
                xt['input_ids'], xt['attention_mask'], xt['token_type_ids'] = ds['input_ids'][trn_idxs], ds['attention_mask'][trn_idxs], ds['token_type_ids'][trn_idxs]
                yt = target[trn_idxs]
                xs['input_ids'], xs['attention_mask'], xs['token_type_ids'] = ds['input_ids'][val_idxs], ds['attention_mask'][val_idxs], ds['token_type_ids'][val_idxs]
                ys = target[val_idxs]
                self.ds.printMemoryUssage('train feature x y')
                #
                self.xtrain, self.ytrain = self.getFeatureLM(xt, yt)
                self.xtest, self.ytest = self.getFeatureLM(xs, ys)
            else:
                self.xtrain, self.ytrain = xtrain, ytrain
                self.xtest, self.ytest = xtest, ytest
            name = self.name
            self.logger.info(f"train {name} model")
            self.model[name] = self.buildModel(mtype=name)
            self.fitModel(name=name, mtype=name, X=self.xtrain, Y=self.ytrain)

            self.writeModel(name)
            
            py = self.predict(self.xtrain)
            tloss = log_loss(self.ytrain, py)

            pyt = self.predict(self.xtest)
            tstloss = log_loss(self.ytest, pyt)
            self.logger.info(f'#################### model {name} training log loss = {tloss} test log loss = {tstloss}')
        except:
            traceback.print_exc()
            self.logger.warning(f'train error' )
        gc.collect()

# end class ModelGeneric
########################################
# end class ModelGeneric
########################################

########################################
class ModelNN(ModelBase):
    def __init__(self, dataSource=None, name='model', opt=None):
        super(ModelNN, self).__init__(dataSource=dataSource, name=name, opt=opt)
        self.trainEpoch = USE_MODEL_NN_TRAIN_EPOCH
        self.trainer = None

    ################
    def setTrainEpoch(self, epoch=10):
        self.trainEpoch = epoch

    ################
    def buildModel(self, mtype=USE_MODEL_PATH, num_labels=3, opt=None):
        if (opt is None):
            opt = {};
        #
        model = AutoModelForSequenceClassification.from_pretrained(mtype, num_labels=num_labels)
        self.logger.info(f'Model = {mtype} opt = {str(opt)}')
        return model

    ################
    def predict(self, xt):
        try:
            y = self.model[self.name](**xt)
            y = y.logits.detach().numpy()
            return y
        except:
            self.logger.warning(f'predict error' ) ;
            traceback.print_exc()
            return 0

    ################
    def readModel(self, name, num_labels=3, opt=None):
        self.name = name
        self.model[name] = AutoModelForSequenceClassification.from_pretrained(USE_MODEL_PATH_SAVE, num_labels=num_labels)
        self.logger.info(f'Model = {name} opt = {str(opt)}')

    ################
    # train on all data(split train/test)
    def train(self) :
        from sklearn.metrics import log_loss
        import torch.nn.functional as F
        def score(preds): return {'log loss': log_loss(preds.label_ids, F.softmax(torch.Tensor(preds.predictions)))}
        
        self.ds.printMemoryUssage('train')
        trn_idxs, val_idxs = self.ds.getTrainTestSplit(0.2)
        ds, target = self.getFeatureDS()
        #dds = DatasetDict({"train":ds.select(trn_idxs), "test": ds.select(val_idxs)})
        trn = {}
        tst = {}
        trn['input_ids'], trn['attention_mask'], trn['token_type_ids'], trn['label'] = ds['input_ids'][trn_idxs], ds['attention_mask'][trn_idxs], ds['token_type_ids'][trn_idxs], target[trn_idxs]
        tst['input_ids'], tst['attention_mask'], tst['token_type_ids'], tst['label'] = ds['input_ids'][trn_idxs], ds['attention_mask'][trn_idxs], ds['token_type_ids'][trn_idxs], target[trn_idxs]
        self.ds.printMemoryUssage('train feature x y')
        #
        self.logger.info(f"Train {self.name} model")
        name = self.name;
        self.model[name] = self.buildModel(mtype=USE_MODEL_PATH, num_labels=3)
        lr,bs = 8e-5,16
        wd,epochs = 0.01, self.trainEpoch
        tokz = AutoTokenizer.from_pretrained(USE_MODEL_PATH)
        args = TrainingArguments('outputs', learning_rate=lr, warmup_ratio=0.1, lr_scheduler_type='cosine', fp16=False,
            evaluation_strategy="epoch", per_device_train_batch_size=bs, per_device_eval_batch_size=bs*2,
            num_train_epochs=epochs, weight_decay=wd, report_to='none')
        dstrn = Dataset.from_dict(trn)
        dstst = Dataset.from_dict(tst)
        self.trainer = Trainer(self.model[name], args, train_dataset=dstrn, eval_dataset=dstst,
                            tokenizer=tokz, compute_metrics=score)        
        try:
            self.trainer.train()
            # mode saved

            self.model[name].save_pretrained(USE_MODEL_PATH_SAVE)
            
            self.logger.info(f'model {self.name} ')
        except: 
            traceback.print_exc()

# end class ModelNN
########################################