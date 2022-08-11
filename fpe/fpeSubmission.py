########################################
##
import pandas as pd
import numpy as np
import logging
import traceback
import os.path
import os
import pickle
import time
import datetime
from datetime import datetime
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import psutil
import gc
from scipy.stats import pearsonr

from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor
from sklearn import linear_model
from xgboost import XGBRegressor
from xgboost.sklearn import XGBRFRegressor
from xgboost.sklearn import XGBRFClassifier
from lightgbm import LGBMClassifier
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

########################################
from fpe import *
from fpeDataSource import *
from fpeModel import *

########################################
# CONSTANT and setups
DO_SUBMISSION = 1 ; # = 0|None: submit real; 1 : local simulate; otherwise : DO_ nothing

####################################
class ModelSubmission(ModelBase):
    def __init__(self, dataSource=None, name='model', opt=None):
        super(ModelSubmission, self).__init__(dataSource=dataSource, name=name, opt=opt)
        # model
        if (USE_MODEL_LGB):
            self.modelLGB = ModelGeneric(dataSource=dataSource,name=MODEL_NAME_LGB, opt=opt) ;
        if (USE_MODEL_XGB):
            self.modelXGB = ModelGeneric(dataSource=dataSource,name=MODEL_NAME_XGB, opt=opt) ;
        if (USE_MODEL_NN):
            self.modelNN = ModelNN(dataSource=dataSource,name=MODEL_NAME_DNN, opt=opt) ;

        self.loadModels() ;

    ################
    # load all models
    def loadModels(self):
        if (USE_MODEL_LGB):
            self.modelLGB.readModel(MODEL_NAME_LGB)
        if (USE_MODEL_XGB):
            self.modelXGB.readModel(MODEL_NAME_XGB)
        if (USE_MODEL_NN):
            self.modelNN.readModel(MODEL_NAME_DNN)

    def getFeature(self):
        try:
            ds = DataSource(dataPath=USE_DATA_PATH)
            ds.printMemoryUssage('readDataSource')
            ds.readData(fileName=USE_TEST_FILENAME)
            enx, _ = ds.getFeature(hasTarget=False)
            hidden = None
            if USE_MODEL_LGB or USE_MODEL_XGB:
                if USE_MODEL_LGB:
                    hidden, _ = self.modelLGB.getFeatureLM(enx, None, hasTarget=False)
                else:
                    hidden, _ = self.modelXGB.getFeatureLM(enx, None, hasTarget=False)

            return enx, hidden
        except:
            traceback.print_exc()

    ################
    def predict(self, enx, hidden):
        #import pdb; pdb.set_trace()
        yp = np.zeros((enx['input_ids'].shape[0],3), dtype=np.float32)
        wt = 0 ;
        if (USE_MODEL_LGB):
            yp = yp + self.modelLGB.predict( hidden ) * 0.2
            wt += 0.2 ;
        if (USE_MODEL_XGB):
            yp = yp + self.modelXGB.predict( hidden ) * 0.2
            wt += 0.2 ;
        if (USE_MODEL_NN):
            yp = yp + self.modelNN.predict( enx ) * 0.6
            wt += 0.6
        ypred = yp / wt ;
        return ypred;

    ################
    ## submission
    def submit(self, simulate=0):
        self.logger.info(f'Submission')
        logging.getLogger('fpe').warning(f'submission start')
        enx, hidden = self.getFeature()
        preds = self.predict(enx, hidden)
        submission_df = pd.read_csv(USE_DATA_PATH + '/sample_submission.csv')
        submission_df['Ineffective'] = preds[:,0]
        submission_df['Adequate'] = preds[:,1]
        submission_df['Effective'] = preds[:,2]
        submission_df.to_csv('submission.csv',index=False)
        logging.getLogger('fpe').warning(f'submission end')
        #import pdb; pdb.set_trace()
        self.logger.info(f'End submission')
########################################

########################################
# submit
setupLogger( );
logging.getLogger('fpe').info(f'start')
fpeMS = ModelSubmission()
fpeMS.submit()
logging.getLogger('fpe').info(f'end')
