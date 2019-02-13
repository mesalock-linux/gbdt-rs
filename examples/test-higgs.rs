extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::{Data, DataVec, PredVec, ValueType};
use gbdt::fitness::almost_equal_thrs;
use gbdt::gradient_boost::GBDT;

use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {
    let mut cfg = Config::new();
    cfg.set_feature_size(28);
    cfg.set_max_depth(6);
    cfg.set_iterations(10);
    cfg.set_loss("LogLikelyhood");

    // To get the test data, please clone the project from github
    let train_file = "path/to/higgs/train.csv";
    let test_file = "path/to/higgs/test.csv";

    let mut train_dv: DataVec = Vec::new();
    let mut test_dv: DataVec = Vec::new();

    let f = File::open(train_file).unwrap();
    let f = BufReader::new(f);
    for line in f.lines() {
        let l = line.unwrap();
        let lv: Vec<&str> = l.splitn(29, ",").collect();
        assert!(lv.len() == 29);
        let mut feature: Vec<ValueType> = Vec::new();
        for i in 1..29 {
            feature.push(lv[i].parse::<ValueType>().unwrap());
        }
        let d = Data {
            feature: feature,
            target: 0.0,
            weight: 1.0,
            label: lv[0].parse::<ValueType>().unwrap(),
            residual: 0.0,
            initial_guess: 0.0,
        };
        train_dv.push(d);
    }

    let f = File::open(test_file).unwrap();
    let f = BufReader::new(f);
    for line in f.lines() {
        let l = line.unwrap();
        let lv: Vec<&str> = l.splitn(29, ",").collect();
        assert!(lv.len() == 29);
        let mut feature: Vec<ValueType> = Vec::new();
        for i in 1..29 {
            feature.push(lv[i].parse::<ValueType>().unwrap());
        }
        let d = Data {
            feature: feature,
            target: 0.0,
            weight: 1.0,
            label: lv[0].parse::<ValueType>().unwrap(),
            residual: 0.0,
            initial_guess: 0.0,
        };
        test_dv.push(d);
    }

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
        println!("[{}]  {}  {}", i, test_dv[i].label, predicted[i]);
    }

    println!("correct: {}", correct);
    println!("wrong:   {}", wrong);
}
