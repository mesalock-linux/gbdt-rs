extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::{Data, DataVec, PredVec, ValueType, VALUE_TYPE_UNKNOWN};
use gbdt::fitness::almost_equal_thrs;
use gbdt::gradient_boost::GBDT;

use std::fs::File;
use std::io::{BufRead, BufReader};
use time::PreciseTime;

fn main() {
    println!("Start test allstate");
    let mut cfg = Config::new();
    cfg.set_feature_size(35);
    cfg.set_max_depth(6);
    cfg.set_iterations(50);
    cfg.set_loss("SquaredError");
    cfg.set_data_sample_ratio(0.7);
    //cfg.set_feature_sample_ratio(0.7);
    cfg.set_min_leaf_size(3);

    let train_file = "path/to/allstate/train.csv";
    let test_file = "path/to/allstate/test.csv";

    let mut train_dv: DataVec = Vec::new();
    let mut test_dv: DataVec = Vec::new();

    let t1 = PreciseTime::now();

    let f = File::open(train_file).unwrap();
    let f = BufReader::new(f);
    let mut flag = 0;
    for line in f.lines() {
        if flag == 0 {
            flag = 1;
            continue;
        }
        let l = line.unwrap();
        let lv: Vec<&str> = l.splitn(36, ",").collect();
        assert!(lv.len() == 36);
        let mut feature: Vec<ValueType> = Vec::new();
        for i in 0..35 {
            feature.push(match lv[i].parse::<ValueType>() {
                Ok(num) => num,
                Err(_error) => VALUE_TYPE_UNKNOWN,
            });
        }
        let d = Data {
            feature: feature,
            target: 0.0,
            weight: 1.0,
            label: lv[35].parse::<ValueType>().unwrap(),
            residual: 0.0,
            initial_guess: 0.0,
        };
        train_dv.push(d);
    }

    let f = File::open(test_file).unwrap();
    let f = BufReader::new(f);
    let mut flag = 0;
    for line in f.lines() {
        if flag == 0 {
            flag = 1;
            continue;
        }
        let l = line.unwrap();
        let lv: Vec<&str> = l.splitn(36, ",").collect();
        assert!(lv.len() == 36);
        let mut feature: Vec<ValueType> = Vec::new();
        for i in 0..35 {
            feature.push(match lv[i].parse::<ValueType>() {
                Ok(num) => num,
                Err(_error) => VALUE_TYPE_UNKNOWN,
            });
        }
        let d = Data {
            feature: feature,
            target: 0.0,
            weight: 1.0,
            label: lv[35].parse::<ValueType>().unwrap(),
            residual: 0.0,
            initial_guess: 0.0,
        };
        test_dv.push(d);
    }

    let t2 = PreciseTime::now();
    println!("Load data: {}", t1.to(t2));

    let mut gbdt = GBDT::new(&cfg);
    gbdt.fit(&mut train_dv);
    let predicted: PredVec = gbdt.predict(&test_dv);

    assert_eq!(predicted.len(), test_dv.len());
    let mut correct = 0;
    let mut wrong = 0;

    for i in 0..predicted.len() {
        if almost_equal_thrs(test_dv[i].label, predicted[i], 0.2) {
            correct += 1;
        } else {
            wrong += 1;
        };
    }

    println!("correct: {}", correct);
    println!("wrong:   {}", wrong);
}
