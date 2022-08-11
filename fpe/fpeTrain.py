########################################
##
from fpe import *
from fpeDataSource import *
from fpeModel import *

########################################
# train
setupLogger( );
logging.getLogger('fpe').info(f'start training')

ds = DataSource(dataPath=USE_DATA_PATH)
ds.printMemoryUssage('readDataSource')
ds.readData(fileName=USE_TRAIN_FILENAME)
ds.printMemoryUssage('readDataSource Done')

if USE_MODEL_NN:
    dnn = ModelNN (dataSource=ds,name=MODEL_NAME_DNN)
    dnn.train()

if USE_MODEL_LGB:
    lgm = ModelGeneric (dataSource=ds,name=MODEL_NAME_LGB)
    lgm.train()
if USE_MODEL_XGB:
    xgb = ModelGeneric (dataSource=ds,name=MODEL_NAME_XGB)
    if USE_MODEL_LGB:
        xgb.train(xtrain=lgm.xtrain,ytrain=lgm.ytrain,xtest=lgm.xtest,ytest=lgm.ytest)
    else:
        xgb.train()
        
logging.getLogger('fpe').info(f'End train execution')
