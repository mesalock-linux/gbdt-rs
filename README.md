# MesaTEE GBDT-RS

[![Build Status](https://ci.mesalock-linux.org/api/badges/mesalock-linux/gbdt-rs/status.svg)](https://ci.mesalock-linux.org/mesalock-linux/gbdt-rs)
[![codecov](https://codecov.io/gh/mesalock-linux/gbdt-rs/branch/master/graph/badge.svg)](https://codecov.io/gh/mesalock-linux/gbdt-rs)

MesaTEE GBDT-RS is a gradient boost decision tree library written in Safe Rust. There is no unsafe rust code in the library. 

MesaTEE GBDT-RS provides the training and inference capabilities. And it can use the models trained by [xgboost](https://xgboost.readthedocs.io/en/latest/) to do inference tasks.

New! The MesaTEE GBDT-RS [paper](gbdt.pdf) has been [accepted by IEEE S&P'19](https://www.ieee-security.org/TC/SP2019/program-posters.html)!


# Supported Task
## Supppoted task for both training and inference
1. Linear regression: use SquaredError and LAD loss types 
2. Binary classification (labeled with 1 and -1): use LogLikelyhood loss type
## Compatibility with xgboost 
At this time, MesaTEE GBDT-RS support to use model trained by xgboost to do inference.  The model should be trained by xgboost with following configruation:

1. booster: gbtree
2. objective: "reg:linear", "reg:logistic", "binary:logistic", "binary:logitraw", "multi:softprob", "multi:softmax" or "rank:pairwise".

We have tested that MesaTEE GBDT-RS is compatible with xgboost 0.81 and 0.82

# Quick Start
## Training Steps
1. Set configuration
2. Load training data
3. Train the model
4. (optional) Save the model

## Inference Steps
1. Load the model
2. Load the test data
3. Inference the test data

## Example
``` rust
    use gbdt::config::Config;
    use gbdt::decision_tree::{DataVec, PredVec};
    use gbdt::gradient_boost::GBDT;
    use gbdt::input::{InputFormat, load};

    let mut cfg = Config::new();
    cfg.set_feature_size(22);
    cfg.set_max_depth(3);
    cfg.set_iterations(50);
    cfg.set_shrinkage(0.1);
    cfg.set_loss("LogLikelyhood"); 
    cfg.set_debug(true);
    cfg.set_data_sample_ratio(1.0);
    cfg.set_feature_sample_ratio(1.0);
    cfg.set_training_optimization_level(2);

    // load data
    let train_file = "dataset/agaricus-lepiota/train.txt";
    let test_file = "dataset/agaricus-lepiota/test.txt";

    let mut input_format = InputFormat::csv_format();
    input_format.set_feature_size(22);
    input_format.set_label_index(22);
    let mut train_dv: DataVec = load(train_file, input_format).expect("failed to load training data");
    let test_dv: DataVec = load(test_file, input_format).expect("failed to load test data");

    // train and save model
    let mut gbdt = GBDT::new(&cfg);
    gbdt.fit(&mut train_dv);
    gbdt.save_model("gbdt.model").expect("failed to save the model");

    // load model and do inference
    let model = GBDT::load_model("gbdt.model").expect("failed to load the model");
    let predicted: PredVec = model.predict(&test_dv);
```
## Example code
* Linear regression: examples/iris.rs
*  Binary classification: examples/agaricus-lepiota.rs

# Use models trained by xgboost

## Steps
1. Use xgboost to train a model
2. Use examples/convert_xgboost.py to convert the model
    * Usage: python convert_xgboost.py xgboost_model_path objective output_path
    * Note convert_xgboost.py depends on xgboost python libraries. The converted model can be used on machines without xgboost
3. In rust code, call GBDT::load_from_xgboost(model_path, objective) to load the model
4. Do inference
5. (optional) Call GBDT::save_model to save the model to MesaTEE GBDT-RS native format. 

## Example code
* "reg:linear": examples/test-xgb-reg-linear.rs
* "reg:logistic": examples/test-xgb-reg-logistic.rs
* "binary:logistic": examples/test-xgb-binary-logistic.rs 
* "binary:logitraw": examples/test-xgb-binary-logistic.rs 
* "multi:softprob": examples/test-xgb-multi-softprob.rs
* "multi:softmax": examples/test-xgb-multi-softmax.rs 
* "rank:pairwise": examples/test-xgb-rank-pairwise.rs

# Multi-threading
## Training:
At this time, training in MesaTEE GBDT-RS is single-threaded.
## Inference:
The related inference functions are single-threaded. But they are thread-safe. We provide an inference example using multi threads in example/test-multithreads.rs

# SGX usage
Because MesaTEE GBDT-RS is written in pure rust, with the help of [rust-sgx-sdk](https://github.com/baidu/rust-sgx-sdk), it can be used in sgx enclave easily as:

```
gbdt_sgx = { git = "https://github.com/mesalock-linux/gbdt-rs" }
```

This would import a crate named `gbdt_sgx`. If you prefer `gbdt` as normal:

```
gbdt = { package = "gbdt_sgx", git = "https://github.com/mesalock-linux/gbdt-rs" }
```

For more information and concret examples, please look at directory `sgx/gbdt-sgx-test`.

# License

Apache 2.0

# Authors

Tianyi Li @n0b0dyCN <n0b0dypku@gmail.com>

Tongxin Li @litongxin1991 <litongxin1991@gmail.com>

Yu Ding @dingelish <dingelish@gmail.com>

# Steering Committee
Tao Wei, Yulong Zhang

# Acknowledgment

Thanks to @qiyiping for his/her great previous work [gbdt](https://github.com/qiyiping/gbdt). We read his/her code before starting this project.
