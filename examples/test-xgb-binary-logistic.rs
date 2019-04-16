extern crate gbdt;

use gbdt::decision_tree::{PredVec, ValueType};
use gbdt::gradient_boost::GBDT;
use gbdt::input;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    // Use xg.py in xgb-data/xgb_binary_logistic to generate a model and get prediction results from xgboost.
    // Call this command to convert xgboost model:
    // python examples/convert_xgboost.py xgb-data/xgb_binary_logistic/xgb.model "binary:logistic" xgb-data/xgb_binary_logistic/gbdt.model
    // load model
    let gbdt = GBDT::from_xgoost_dump("xgb-data/xgb_binary_logistic/gbdt.model", "binary:logistic")
        .expect("failed to load model");

    // load test data
    let test_file = "xgb-data/xgb_binary_logistic/agaricus.txt.test";
    let mut input_format = input::InputFormat::txt_format();
    input_format.set_feature_size(126);
    input_format.set_delimeter(' ');
    let test_data = input::load(test_file, input_format).expect("failed to load test data");

    // inference
    println!("start prediction");
    let predicted: PredVec = gbdt.predict(&test_data);
    assert_eq!(predicted.len(), test_data.len());

    // compare to xgboost prediction results
    let predict_result = "xgb-data/xgb_binary_logistic/pred.csv";

    let mut xgb_results = Vec::new();
    let file = File::open(predict_result).expect("failed to load pred.csv");
    let reader = BufReader::new(file);
    for line in reader.lines() {
        let text = line.expect("failed to read data from pred.csv");
        let value: ValueType = text.parse().expect("failed to parse data from pred.csv");
        xgb_results.push(value);
    }

    let mut max_diff: ValueType = -1.0;
    for (value1, value2) in predicted.iter().zip(xgb_results.iter()) {
        println!("{} {}", value1, value2);
        let diff = (value1 - value2).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    println!(
        "Compared to results from xgboost, max error is: {:.10}",
        max_diff
    );
    assert!(max_diff < 0.01);
}
