extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::DataVec;
use gbdt::gradient_boost::GBDT;
use gbdt::input::{load, InputFormat};

fn main() {
    let mut cfg = Config::new();
    cfg.set_feature_size(22);
    cfg.set_max_depth(6);
    cfg.set_iterations(10);
    cfg.set_shrinkage(0.1);
    cfg.set_loss("LogLikelyhood");
    //cfg.set_debug(true);
    //cfg.set_data_sample_ratio(0.8);
    //cfg.set_feature_sample_ratio(0.5);
    cfg.set_training_optimization_level(2);

    // load data
    let train_file = "dataset/agaricus-lepiota/train.txt";

    let mut input_format = InputFormat::csv_format();
    input_format.set_feature_size(22);
    input_format.set_label_index(22);
    let mut train_dv: DataVec =
        load(train_file, input_format).expect("failed to load training data");

    // train and save model
    let mut gbdt = GBDT::new(&cfg);
    gbdt.fit(&mut train_dv);
    gbdt.get_feature_importances();
}
