extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::{Data, DataVec, PredVec, ValueType, VALUE_TYPE_UNKNOWN};
use gbdt::fitness::almost_equal_thrs;
use gbdt::gradient_boost::GBDT;

use std::fs::File;
use std::io::stdin;
use std::io::{BufRead, BufReader};
use time::PreciseTime;

fn main() {
    println!("Start test xgboost");
    let gbdt = GBDT::from_xgoost_dump("data/gbdt.model", "reg:linear").unwrap();
    let test_file = "data/train.csv";
    let f = File::open(test_file).unwrap();
    let f = BufReader::new(f);
    let mut flag = 0;
    let mut test_dv: DataVec = Vec::new();
    for line in f.lines() {
        if flag == 0 {
            flag = 1;
            continue;
        }
        let l = line.unwrap();
        let lv: Vec<&str> = l.splitn(601, ",").collect();
        assert!(lv.len() == 601);
        let mut feature: Vec<ValueType> = Vec::new();
        for i in 0..600 {
            feature.push(match lv[i].parse::<ValueType>() {
                Ok(num) => num,
                Err(_error) => VALUE_TYPE_UNKNOWN,
            });
        }
        let d = Data {
            feature: feature,
            target: 0.0,
            weight: 1.0,
            label: lv[600].parse::<ValueType>().unwrap(),
            residual: 0.0,
            initial_guess: 0.0,
        };
        test_dv.push(d);
    }

    println!("start predict");
    let t1 = PreciseTime::now();
    let predicted: PredVec = gbdt.predict(&test_dv);
    let t2 = PreciseTime::now();
    println!("predict data: {}", t1.to(t2));
    assert_eq!(predicted.len(), test_dv.len());
    let mut correct = 0;
    let mut wrong = 0;
    for i in 0..predicted.len() {
        if almost_equal_thrs(test_dv[i].label, predicted[i], 0.2) {
            correct += 1;
        } else {
            wrong += 1;
        };
        println!("{}", predicted[i]);
        //println!("[{}]  {}  {}", i, test_dv[i].label, predicted[i]);
    }

    println!("correct: {}", correct);
    println!("wrong:   {}", wrong);
}
