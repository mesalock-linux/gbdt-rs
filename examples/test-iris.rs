extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::{DataVec, PredVec};
use gbdt::fitness::almost_equal_thrs;
use gbdt::gradient_boost::GBDT;
use gbdt::input::{InputFormat, load};

fn main() {
    let mut cfg = Config::new();
    cfg.set_feature_size(4);
    cfg.set_max_depth(4);
    cfg.set_iterations(100);
    cfg.set_shrinkage(0.1);
    cfg.set_loss("LAD"); 
    cfg.set_debug(true);
    cfg.set_training_optimization_level(2);

    // load data
    let train_file = "dataset/iris/train.txt";
    let test_file = "dataset/iris/test.txt";

    let mut input_format = InputFormat::csv_format();
    input_format.set_feature_size(4);
    input_format.set_label_index(4);
    let mut train_dv: DataVec = load(train_file, input_format).expect("failed to load training data");
    let test_dv: DataVec = load(test_file, input_format).expect("failed to load test data");

    // train and save the model
    let mut gbdt = GBDT::new(&cfg);
    gbdt.fit(&mut train_dv);
    gbdt.save_model("gbdt.model").expect("failed to save the model");

    // load the model and do inference
    let model = GBDT::load_model("gbdt.model").expect("failed to load the model");
    let predicted: PredVec = model.predict(&test_dv);

    assert_eq!(predicted.len(), test_dv.len());
    let mut correct = 0;
    let mut wrong = 0;
    for i in 0..predicted.len() {
        if almost_equal_thrs(test_dv[i].label, predicted[i], 0.0001) {
            correct += 1;
        } else {
            wrong += 1;
        };
        println!("[{}]  {}  {}", i, test_dv[i].label, predicted[i]);
    }

    println!("correct: {}", correct);
    println!("wrong:   {}", wrong);
}
