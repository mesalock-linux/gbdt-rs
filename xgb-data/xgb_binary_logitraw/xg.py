from __future__ import print_function
import xgboost as xgb
import numpy as np
import time

params = {
    'booster': 'gbtree',
    'objective': 'binary:logitraw',
    'eta': 0.1,
    'gamma': 1.0,
    'max_depth': 3,
    'min_child_weight': 1,
    'seed': 1000,
    'nthread': 1,
}

def train():
    xgb_train = xgb.DMatrix("./agaricus.txt.train")
    plst = list(params.items())
    print("Training started.")
    t0 = time.time()
    model = xgb.train(plst, xgb_train, num_boost_round=50,
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
    xgb_test = xgb.DMatrix("./agaricus.txt.test")
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
