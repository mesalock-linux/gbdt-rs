extern crate gbdt;

use gbdt::decision_tree::PredVec;
use gbdt::gradient_boost::GBDT;
use gbdt::input;

use time::PreciseTime;

fn main() {
    let gbdt =
        GBDT::from_xgoost_dump("data/xgb_binary_logistic/gbdt.model", "binary:logistic").unwrap();
    let test_file = "data/xgb_binary_logistic/agaricus.txt.test";
    let mut input_format = input::InputFormat::txt_format();
    input_format.set_feature_size(126);
    input_format.set_delimeter(' ');
    let test_data = input::load(test_file, input_format);

    println!("start predict");
    let t1 = PreciseTime::now();
    let predicted: PredVec = gbdt.predict(&test_data);
    let t2 = PreciseTime::now();
    println!("predict data: {}", t1.to(t2));
    assert_eq!(predicted.len(), test_data.len());
    for i in 0..predicted.len() {
        println!("{:.10}", predicted[i]);
    }
}