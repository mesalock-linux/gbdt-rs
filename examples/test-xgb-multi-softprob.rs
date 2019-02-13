extern crate gbdt;

use gbdt::decision_tree::{Data, DataVec, ValueType, VALUE_TYPE_UNKNOWN};
use gbdt::gradient_boost::GBDT;
use gbdt::input::{load, InputFormat};

use std::fs::File;
use std::io::{BufRead, BufReader};
use time::PreciseTime;

fn main() {
    let gbdt =
        GBDT::from_xgoost_dump("data/xgb_multi_softmax/gbdt.model", "multi:softmax").unwrap();
    let test_file = "data/xgb_multi_softmax/dermatology.data.test";
    let mut fmt = InputFormat::csv_format();
    fmt.set_feature_size(34);
    let mut test_data: DataVec = load(test_file, fmt);

    println!("start predict");
    let t1 = PreciseTime::now();
    let (labels, probs) = gbdt.predict_multiclass(&test_data, 6);
    let t2 = PreciseTime::now();
    println!("predict data: {}", t1.to(t2));
    assert_eq!(labels.len(), test_data.len());
    assert_eq!(probs.len(), test_data.len());

    for i in 0..test_data.len() {
        println!("{:?}", probs[i]);
    }
}
