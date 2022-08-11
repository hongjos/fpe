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
import psutil
import gc
from torch.utils.data import DataLoader
import warnings,transformers,logging,torch
from transformers import TrainingArguments,Trainer
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from transformers import AutoModel
import datasets
from datasets import load_dataset, Dataset, DatasetDict

from fpe import *

_MEM_VERBOSITY = 1

########################################
# log
def setupLogger(name='fpe.log')->None:
    logger = logging.getLogger('fpe')
    if ('DO_SUBMISSION' in globals() and DO_SUBMISSION == 0):
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.INFO)
    # create file handler which logs even debug messages
    h = logging.FileHandler(name, mode='a')
    h.setLevel(logging.INFO)
    fh = logging.StreamHandler()
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # create formatter and add it to the handlers
    #formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    #formatter = logging.Formatter('%(levelname)s - %(message)s')
    formatter = logging.Formatter('%(message)s')
    formatterDetail = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    h.setFormatter(formatter)
    fh.setFormatter(formatter)
    ch.setFormatter(formatterDetail)
    # add the handlers to the logger
    logger.addHandler(h)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f'') # an empty log item
    logger.info(f'')

########################################
# data source class
class DataSource(object):
    ################
    ################
    def __init__(self, dataPath=None):
        self.logger = logging.getLogger('fpe.source')
        self.dataPath=dataPath;
        if (self.dataPath is None):
            self.dataPath = "/kaggle/input/feedback-prize-effectiveness/"
        self.raw = pd.DataFrame();
        self.sid = []

    ################
    # setPath - set dataPath
    def setPath(self, dataPath ):
        self.dataPath = dataPath;

    ################
    # setup a dataset of empty
    def setupDataEmpty(self):
        self.raw = pd.DataFrame();

    ################
    # setup a dataset of empty
    def gcCollect(self, wait=5):
        gc.collect()
        time.sleep(wait)

    ################
    def printMemoryUssage(self, strInfo='', verbose=0):
        if (_MEM_VERBOSITY >= verbose):
            rmem = psutil.Process().memory_info().rss / (1024 * 1024 * 1024)
            self.logger.info( f'{strInfo} resident memory = {rmem:.2f}G')

    ################
    # read a dataset from training, clean data, split into asset df
    def readData(self, fileName='train.csv', dataPath=None):
        if (not dataPath is None):
            self.setPath(dataPath)
        # initialize to empty df
        self.setupDataEmpty()
        # read from file
        if (not isinstance(fileName, list)):
            fileList = [fileName]
        else:
            fileList = fileName

        dflist = []
        for name in fileList:
            self.logger.info(f'Load data {name}')
            if (name.lower().endswith('.gzip')) or (name.lower().endswith('.parquet')):
                dflist.append(pd.read_parquet(self.dataPath + name))
            else:
                dflist.append(pd.read_csv(self.dataPath + name))
        self.raw = pd.concat(dflist)
        # preprocess data
        self.gcCollect()
        mems = self.raw.memory_usage().sum()
        self.logger.info( f'Memory size = {mems/1000_0000:.0f}M' )
        self.printMemoryUssage( 'readData Done')
        self.logger.info(f'Load data Done')

    ################
    # getDataRange
    def getDataRange(self, start=None, end=None):
        st = 0
        if (not start is None):
            rows = self.raw.shape[0]
            st = min(max(start, 0), rows)

        en = self.raw.shape[0]
        if (not end is None):
            rows = self.raw.shape[0]
            en = min(max(end, 0), rows)
        return st, en

    ################
    # getData
    def getData(self, start=None, end=None ):
        st, ed = self.getDataRange( start, end )
        if (st == 0) and (ed == self.raw.shape[0]):
            return self.raw
        else:
            return self.raw.iloc[st:ed]

    ################
    # computeFeature
    def computeFeature(self, dfraw, hasTarget=True):
        # feature dimension
        tokz = AutoTokenizer.from_pretrained(USE_MODEL_PATH)
        #
        df = dfraw.copy()

        if hasTarget:
            new_label = {"discourse_effectiveness": {"Ineffective": 0, "Adequate": 1, "Effective": 2}}
            df = df.replace(new_label)
            df = df.rename(columns = {"discourse_effectiveness": "label"})        

        sep = tokz.sep_token
        df['inputs'] = df.discourse_type + sep + df.discourse_text

        ds = Dataset.from_pandas(df)
        inps = "discourse_text","discourse_type"
        #def tok_func(x): return tokz(x["inputs"], padding=True, truncation=True, return_tensors='pt')
        #tok_ds = ds.map(tok_func, batched=True, remove_columns=inps+('inputs','discourse_id','essay_id'))        
        tok_ds = tokz(df["inputs"].tolist(), max_length=512, padding=True, truncation=True, return_tensors='pt')

        if hasTarget:
            target = df['label'].values
        else:
            target = []

        return tok_ds, target

    ################
    def getFeature(self, start=None, end=None, hasTarget=True):
        df = self.getData(start=start, end=end)
        self.printMemoryUssage('getFeature', verbose=1)
        tk_ds, target = self.computeFeature( df, hasTarget )
        return tk_ds, target

    ################
    def getTrainTestSplit(self, split=0.2):
        essay_ids = self.raw.essay_id.unique()
        np.random.seed(42)
        np.random.shuffle(essay_ids)
        val_prop = split
        val_sz = int(len(essay_ids)*val_prop)
        val_essay_ids = essay_ids[:val_sz]
        is_val = np.isin(self.raw.essay_id, val_essay_ids)
        idxs = np.arange(len(self.raw))
        val_idxs = idxs[ is_val]
        trn_idxs = idxs[~is_val]        
        return trn_idxs, val_idxs
# end class DataSource
########################################



