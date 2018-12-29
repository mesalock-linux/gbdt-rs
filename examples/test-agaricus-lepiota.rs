extern crate gbdt;

use gbdt::decision_tree::{DataVec, PredVec, Data, ValueType};
use gbdt::config::{Config, Loss};
use gbdt::gradient_boost::GBDT;
use gbdt::fitness::almost_equal;

use std::fs::{File};
use std::io::{BufReader, BufRead};

fn main() {
    let cfg = Config {
        number_of_feature: 22,
        max_depth: 5,
        iterations: 5,
        shrinkage: 1.0,
        feature_sample_ratio: 1.0,
        data_sample_ratio: 1.0,
        min_leaf_size: 0,
        loss: Loss::LogLikelyhood,
        debug: false,
        feature_cost: Vec::new(),
        enable_feature_tunning: false,
        enable_initial_guess: false,
    };

    let train_file = "/usr/src/myapp/dataset/agaricus-lepiota/train.txt";
    let test_file = "/usr/src/myapp/dataset/agaricus-lepiota/test.txt";

    let mut train_dv: DataVec = Vec::new();
    let mut test_dv: DataVec = Vec::new();

    let f = File::open(train_file).unwrap();
    let f = BufReader::new(f);
    for line in f.lines() {
        let l = line.unwrap();
        let lv: Vec<&str> = l.splitn(23, ",").collect();
        assert!(lv.len()==23);
        let mut feature: Vec<ValueType> = Vec::new();
        for i in 0..22 {
            feature.push(lv[i].parse::<ValueType>().unwrap());
        }
        let d = Data {
            feature: feature,
            target: 0.0,
            weight: 1.0,
            label: lv[4].parse::<f64>().unwrap(),
            residual: 0.0,
            initial_guess: 0.0,
        };
        train_dv.push(d);
    }

    let f = File::open(test_file).unwrap();
    let f = BufReader::new(f);
    for line in f.lines() {
        let l = line.unwrap();
        let lv: Vec<&str> = l.splitn(23, ",").collect();
        assert!(lv.len()==23);
        let mut feature: Vec<ValueType> = Vec::new();
        for i in 0..22 {
            feature.push(lv[i].parse::<ValueType>().unwrap());
        }
        let d = Data {
            feature: feature,
            target: 0.0,
            weight: 1.0,
            label: lv[4].parse::<f64>().unwrap(),
            residual: 0.0,
            initial_guess: 0.0,
        };
        test_dv.push(d);
    }

    let mut gbdt = GBDT::new(&cfg);
    gbdt.fit(&train_dv);
    let predicted: PredVec = gbdt.predict(&test_dv);

    assert_eq!(predicted.len(), test_dv.len());
    let mut correct = 0;
    let mut wrong = 0;
    for i in 0..predicted.len() {
        match almost_equal(test_dv[i].label, predicted[i]) {
            true => { correct += 1; },
            false => { wrong += 1; },
        };
    }

    println!("correct: {}", correct);
    println!("wrong:   {}", wrong);

}