extern crate gbdt_sgx;
use gbdt_sgx::config::Config;
use gbdt_sgx::decision_tree::{DataVec, PredVec};
use gbdt_sgx::fitness::AUC;
use gbdt_sgx::gradient_boost::GBDT;
use gbdt_sgx::input::{load, InputFormat};

pub fn test_main() {
    let mut cfg = Config::new();
    cfg.set_feature_size(22);
    cfg.set_max_depth(3);
    cfg.set_iterations(50);
    cfg.set_shrinkage(0.1);
    cfg.set_loss("LogLikelyhood");
    cfg.set_debug(true);
    //cfg.set_data_sample_ratio(0.8);
    //cfg.set_feature_sample_ratio(0.5);
    cfg.set_training_optimization_level(2);

    // load data
    let train_file = "dataset/agaricus-lepiota/train.txt";
    let test_file = "dataset/agaricus-lepiota/test.txt";

    let mut input_format = InputFormat::csv_format();
    input_format.set_feature_size(22);
    input_format.set_label_index(22);
    let mut train_dv: DataVec =
        load(train_file, input_format).expect("failed to load training data");
    let test_dv: DataVec = load(test_file, input_format).expect("failed to load test data");

    // train and save model
    let mut gbdt = GBDT::new(&cfg);
    gbdt.fit(&mut train_dv);
    gbdt.save_model("gbdt.model")
        .expect("failed to save the model");

    // load model and do inference
    let model = GBDT::load_model("gbdt.model").expect("failed to load the model");
    let predicted: PredVec = model.predict(&test_dv);

    assert_eq!(predicted.len(), test_dv.len());
    let mut correct = 0;
    let mut wrong = 0;
    for i in 0..predicted.len() {
        let label = if predicted[i] > 0.5 { 1.0 } else { -1.0 };
        if (test_dv[i].label - label).abs() < 0.0001 {
            correct += 1;
        } else {
            wrong += 1;
        };
        //println!("[{}]  {}  {}", i, test_dv[i].label, predicted[i]);
    }

    println!("correct: {}", correct);
    println!("wrong:   {}", wrong);

    let auc = AUC(&test_dv, &predicted, test_dv.len());
    println!("AUC: {}", auc);

    use gbdt_sgx::fitness::almost_equal;
    assert_eq!(wrong, 0);
    assert!(almost_equal(auc, 1.0));
}
