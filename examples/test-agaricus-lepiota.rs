extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::{DataVec, PredVec};
use gbdt::fitness::almost_equal_thrs;
use gbdt::gradient_boost::GBDT;
use gbdt::input::{infer, load};

fn main() {
    let mut cfg = Config::new();
    cfg.set_feature_size(22);
    cfg.set_max_depth(5);
    cfg.set_iterations(5);
    cfg.set_loss("SquaredError");

    // To get the test data, please clone the project from github
    let train_file = "dataset/agaricus-lepiota/train.txt";
    let test_file = "dataset/agaricus-lepiota/test.txt";

    let mut input_format = infer(train_file);
    input_format.set_label_index(22);
    let mut train_dv: DataVec = load(train_file, input_format).unwrap();
    let test_dv: DataVec = load(test_file, input_format).unwrap();

    let mut gbdt = GBDT::new(&cfg);
    gbdt.fit(&mut train_dv);
    let predicted: PredVec = gbdt.predict(&test_dv);

    assert_eq!(predicted.len(), test_dv.len());
    let mut correct = 0;
    let mut wrong = 0;
    for i in 0..predicted.len() {
        if almost_equal_thrs(test_dv[i].label, predicted[i], 0.02) {
            correct += 1;
        } else {
            wrong += 1;
        };
        println!("[{}]  {}  {}", i, test_dv[i].label, predicted[i]);
    }

    println!("correct: {}", correct);
    println!("wrong:   {}", wrong);
}
