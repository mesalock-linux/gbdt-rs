extern crate gbdt;

use gbdt::config::Config;
use gbdt::decision_tree::{Data, DataVec, PredVec, ValueType, VALUE_TYPE_UNKNOWN};
use gbdt::fitness::almost_equal_thrs;
use gbdt::gradient_boost::GBDT;

use std::fs::File;
use std::io::{BufRead, BufReader};
use time::PreciseTime;

use gbdt::input::InputFormat;
use gbdt::input;
use std::thread;
use std::sync::Arc;

fn main() {
    let thread_num = 12;
    let feature_size = 35;
    let t1 = PreciseTime::now();
    let gbdt =
        GBDT::load_model("./gbdt-rs.model").unwrap();
    let t2 = PreciseTime::now();
    println!("load model: {}", t1.to(t2));
    let t1 = PreciseTime::now();
    let test_file = "./svm.txt";
    let mut fmt = input::InputFormat::txt_format();
    fmt.set_feature_size(feature_size);
    fmt.set_delimeter(' ');
    let mut test_data = input::load(test_file, fmt);
    let t2 = PreciseTime::now();
    println!("load data: {}", t1.to(t2));

    let t1 = PreciseTime::now();
    let mut handles = vec![];
    let mut test_data_vec = vec![];
    let data_size = test_data.len();
    let batch_size = (data_size-1) / thread_num + 1;
    for one_batch in test_data.chunks(batch_size) {
        test_data_vec.push(one_batch.to_vec())

    }

    test_data.clear();
    test_data.shrink_to_fit();
    let t2 = PreciseTime::now();
    println!("split data: {}", t1.to(t2));


    let t1 = PreciseTime::now();
    let gbdt_arc = Arc::new(gbdt);
    for data in test_data_vec.into_iter() {
        let gbdt_clone = Arc::clone(&gbdt_arc);
        let handle = thread::spawn(move || {
            gbdt_clone.predict(&data)
        });
        handles.push(handle)
    }


    let mut preds = Vec::with_capacity(data_size);
    for handle in handles {
        preds.append(&mut handle.join().unwrap());
    }

    let t2 = PreciseTime::now();
    println!("predict data: {}", t1.to(t2));
    assert_eq!(preds.len(), data_size);
    
}
