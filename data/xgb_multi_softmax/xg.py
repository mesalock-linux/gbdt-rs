"""
To use this file:
1. Install xgboost, sklearn, pandas and numpy
2. Download dataset from <https://www.kaggle.com/c/ClaimPredictionChallenge> and extract it.
3. Set DATASET_DIR on line 25 to the directory holds the extracted data.
4. set MODEL_DIR on line 31 to the directory you want to save the model.
5. Uncomment the function `pre_work()` in main.
6. Run this scripts.

==========
This script runs about 100 mins on iMac Pro.
"""


import xgboost as xgb
import numpy as np
from  sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import random
import time
import pickle

params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',
    'eta': 0.1,
    'max_depth': 6,
    'num_class': 6,
    'nthread': 1,
}

def train():
    data = np.loadtxt('./dermatology.data.train', delimiter=',',
        converters={33: lambda x:int(x == '?'), 34: lambda x:int(x) - 1})
    train_X = data[:, :33]
    train_Y = data[:, 34]

    xgb_train = xgb.DMatrix(train_X, label=train_Y)
    plst = list(params.items())
    print("Training started.")
    t0 = time.time()
    model = xgb.train(plst, xgb_train, num_boost_round=5,
    )
    print("%.3fs taken for training" % (time.time() - t0))
    print("Saving model...")
    model.save_model("xgb.model")

def predict():
    t0 = time.time()
    model = xgb.Booster()
    model.load_model("xgb.model")
    t1 = time.time()
    print("%.3fs taken for load_model" % (t1 - t0))

    t0 = time.time()
    data = np.loadtxt('./dermatology.data.test', delimiter=',',
        converters={33: lambda x:int(x == '?'), 34: lambda x:int(x) - 1})
    test_X = data[:, :33]
    test_Y = data[:, 34]

    xgb_test = xgb.DMatrix(test_X, label=test_Y)
    t1 = time.time()
    print("%.3fs taken for load_data" % (t1 - t0))

    t0 = time.time()
    preds = model.predict(xgb_test)
    t1 = time.time()
    print("%.3fs taken for predicting" % (t1 - t0))

    print("Saving results...")
    np.savetxt("./pred.csv", preds, delimiter=",")


if __name__ == "__main__":
    train()
    predict()
