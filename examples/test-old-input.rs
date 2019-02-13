use gbdt::decision_tree::{DataVec, Data, ValueType};
use time::PreciseTime;
use std::fs::File;
use std::io::{BufRead, BufReader};
fn main() {
    let start = PreciseTime::now();
    let train_file = "/Users/icst/workspace/xgboost/dataset/higgs/train.csv";
    let mut train_dv: DataVec = Vec::new();
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
    let end = PreciseTime::now();
    println!("Old input time: {}", start.to(end));
}
