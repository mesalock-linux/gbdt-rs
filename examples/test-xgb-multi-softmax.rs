extern crate gbdt;

use gbdt::decision_tree::{Data, DataVec, ValueType, VALUE_TYPE_UNKNOWN};
use gbdt::gradient_boost::GBDT;

use std::fs::File;
use std::io::{BufRead, BufReader};
use time::PreciseTime;

fn main() {
    let gbdt =
        GBDT::from_xgoost_dump("data/xgb_multi_softmax/gbdt.model", "multi:softmax").unwrap();
    let test_file = "data/xgb_multi_softmax/dermatology.data.test";
    let f = File::open(test_file).unwrap();
    let f = BufReader::new(f);
    let mut test_data: DataVec = Vec::new();
    for line in f.lines() {
        let l = line.unwrap();
        let lv: Vec<&str> = l.splitn(35, ",").collect();
        assert!(lv.len() == 35);
        let mut feature: Vec<ValueType> = Vec::new();
        for i in 0..33 {
            feature.push(match lv[i].parse::<ValueType>() {
                Ok(num) => num,
                Err(_error) => VALUE_TYPE_UNKNOWN,
            });
        }
        let d = Data {
            feature: feature,
            target: 0.0,
            weight: 1.0,
            label: lv[34].parse::<ValueType>().unwrap(),
            residual: 0.0,
            initial_guess: 0.0,
        };
        test_data.push(d);
    }

    println!("start predict");
    let t1 = PreciseTime::now();
    let (labels, probs) = gbdt.predict_multiclass(&test_data, 6);
    let t2 = PreciseTime::now();
    println!("predict data: {}", t1.to(t2));
    assert_eq!(labels.len(), test_data.len());
    assert_eq!(probs.len(), test_data.len());

    for i in 0..test_data.len() {
        println!("{:?}", labels[i]);
    }
}
