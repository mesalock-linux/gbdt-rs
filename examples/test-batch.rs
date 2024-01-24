extern crate gbdt;

use gbdt::decision_tree::ValueType;
use gbdt::gradient_boost::GBDT;
use gbdt::input;
use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    // Use xg.py in xgb-data/xgb_reg_linear to generate a model and get prediction results from xgboost.
    // Call this command to convert xgboost model:
    // python examples/convert_xgboost.py xgb-data/xgb_reg_linear/xgb.model "reg:linear" xgb-data/xgb_reg_linear/gbdt.model
    // load model
    let gbdt = GBDT::from_xgboost_dump("xgb-data/xgb_reg_linear/gbdt.model", "reg:linear")
        .expect("failed to load model");

    // load test data
    let test_file = "xgb-data/xgb_reg_linear/machine.txt.test";
    let mut input_format = input::InputFormat::txt_format();
    input_format.set_feature_size(36);
    input_format.set_delimeter(' ');
    let test_data = input::load(test_file, input_format).expect("failed to load test data");

    // inference
    println!("start prediction");
    let mut predicted = Vec::with_capacity(test_data.len());
    for (count, data) in test_data.chunks(12).enumerate() {
        println!("batch {}: size {}", count, data.len());
        let mut predicted_batch = gbdt.predict(&data.to_vec());
        predicted.append(&mut predicted_batch);
    }
    assert_eq!(predicted.len(), test_data.len());

    // compare to xgboost prediction results
    let predict_result = "xgb-data/xgb_reg_linear/pred.csv";

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
