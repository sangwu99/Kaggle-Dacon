import pandas as pd 

TRAIN = pd.read_csv('data/train_data.csv',encoding='utf-8')
TEST = pd.read_csv('data/test_data.csv',encoding='utf-8')
EVAL_KLUE = pd.read_json("data/klue-nli-v1.1_dev.json")
TRAIN_KLUE = pd.read_json("data/klue-nli-v1.1_train.json")

MODEL_NAME = 'tunib/electra-ko-base'

