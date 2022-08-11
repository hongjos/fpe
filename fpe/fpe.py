USE_MODEL_NN  = True
USE_MODEL_LGB = True
USE_MODEL_XGB = False

DO_SUBMISSION = 1

USE_DATA_PATH = "C:/Temp/Kaggle/feedback-prize-effectiveness/"
if ('DO_SUBMISSION' in globals()) and (DO_SUBMISSION == 0):
    USE_DATA_PATH = "/kaggle/input/feedback-prize-effectiveness/"

USE_TRAIN_FILENAME = ['train.csv']
#USE_TRAIN_FILENAME = ['train2.csv']
USE_TEST_FILENAME = ['test.csv']

MODEL_NAME = 'deberta-v3-small'
MODEL_NAME_SAVE = 'deberta-v3-small-save'

USE_MODEL_PATH = USE_DATA_PATH + "model/" + MODEL_NAME
USE_MODEL_PATH_SAVE = USE_DATA_PATH + "model/" + MODEL_NAME_SAVE

# USE_MODEL_PATH = "microsoft/deberta-v3-small"


USE_MODEL_NN_TRAIN_EPOCH = 1
