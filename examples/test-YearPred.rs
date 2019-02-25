extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::{Data, DataVec, PredVec, ValueType};
use gbdt::fitness::almost_equal_thrs;
use gbdt::gradient_boost::GBDT;
use gbdt::input::{InputFormat, load};

fn main() {
    let mut cfg = Config::new();
    cfg.set_feature_size(90);
    cfg.set_max_depth(20);
    cfg.set_iterations(20);
    cfg.set_loss("SquaredError");

    // To get the test data, please clone the project from github
    let train_file = "/Users/icst/workspace/xgboost/dataset/YearPred/train.csv";
    let test_file = "/Users/icst/workspace/xgboost/dataset/YearPred/test.csv";

    let mut input_format = InputFormat::csv_format();
    let mut train_dv: DataVec = load(train_file, input_format);
    let test_dv: DataVec = load(test_file, input_format);

    let mut gbdt = GBDT::new(&cfg);
    gbdt.fit(&mut train_dv);
    let predicted: PredVec = gbdt.predict(&test_dv);

    assert_eq!(predicted.len(), test_dv.len());
    let mut correct = 0;
    let mut wrong = 0;
    for i in 0..predicted.len() {
        if almost_equal_thrs(test_dv[i].label, predicted[i], 0.4) {
            correct += 1;
        } else {
            wrong += 1;
        };
        println!("[{}]  {}  {}", i, test_dv[i].label, predicted[i]);
    }

    println!("correct: {}", correct);
    println!("wrong:   {}", wrong);
}