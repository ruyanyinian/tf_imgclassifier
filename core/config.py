from easydict import EasyDict as Edict
import os

__C = Edict()

cfg = __C

# data options
__C.DATA = Edict()

__C.DATA.TRAIN_DIR = r"F:\DL_Data\Animal\train_data"
__C.DATA.TEST_DIR = r"F:\DL_Data\Animal\test_data"
# __C.DATA.LABEL_FILENAME = "label.txt"
__C.DATA.TRAIN_OUTPUT_DIR = "./train_tfrecord"
__C.DATA.TEST_OUTPUT_DIR = "./test_tfrecord"
__C.DATA.LABEL_FILEPATH = "./output/label.txt"
__C.DATA.TRAIN_SHARDS = 30
__C.DATA.TEST_SHARDS = 6
__C.DATA.THREADS = 6
__C.DATA.CLASSES = 6

# training options
__C.TRAIN = Edict()

__C.TRAIN.BATCH = 64
__C.TRAIN.EPOCH = 20000
__C.TRAIN.BASE_LR = 0.02

# test options
__C.TEST = Edict()
__C.TEST.BATCH = 1280
__C.TEST.EPOCH = 200

# logs and model options
__C.LOGS = Edict()
__C.LOGS.OUTPUT_DIR = "./output"

# __C.LOGS.MODEL_SAVED_PATH = "./checkpoint/AnimalDector"
__C.LOGS.OUTPUT_SNAPSHOT_NAME = "snapshot"
__C.LOGS.MODEL_SAVED_NAME = "AniDector"
__C.LOGS.INFO_TRAIN = "./output/img_info_train.txt"
__C.LOGS.INFO_TEST = "./output/img_info_test.txt"
__C.LOGS.OUTPUT_SNAPSHOT_DIR = os.path.join(__C.LOGS.OUTPUT_DIR, __C.LOGS.OUTPUT_SNAPSHOT_NAME)
__C.LOGS.MODEL_SAVED_PATH = os.path.join(__C.LOGS.OUTPUT_SNAPSHOT_DIR, __C.LOGS.MODEL_SAVED_NAME)


# image options
__C.IMG = Edict()
__C.IMG.SIZE = 224
__C.IMG.TRAIN_NUMS = 93609
__C.IMG.TEST_NUMS = 3707

# finetune options
__C.FINETUNE = Edict()
__C.FINETUNE.MODEL_LIST = ["mobilenet_v1"]
__C.FINETUNE.MODEL_DIR = r"F:\ImgNet_Pretrained_TF"
__C.FINETUNE.MOBILENET_V1 = os.path.join(__C.FINETUNE.MODEL_DIR, __C.FINETUNE.MODEL_LIST[0])

# display options
__C.DISPLAY = Edict()
__C.DISPLAY.TRAIN_DISPLAY = 1000
__C.DISPLAY.TEST_DISPLAY = 2000


