extern crate gbdt;

use gbdt::gradient_boost::GBDT;

use time::PreciseTime;

use gbdt::input;
use std::sync::Arc;
use std::thread;

fn main() {
    let thread_num = 12;
    let feature_size = 36;
    let model_path = "xgb-data/xgb_reg_linear/gbdt.model";
    let test_file = "xgb-data/xgb_reg_linear/machine.txt.test";

    // load model
    let gbdt = GBDT::from_xgoost_dump(model_path, "reg:linear").expect("faild to load model");

    // load test data
    let mut fmt = input::InputFormat::txt_format();
    fmt.set_feature_size(feature_size);
    fmt.set_delimeter(' ');
    let mut test_data = input::load(test_file, fmt).unwrap();

    // split test data to `thread_num` vectors.
    let t1 = PreciseTime::now();
    let mut handles = vec![];
    let mut test_data_vec = vec![];
    let data_size = test_data.len();
    let batch_size = (data_size - 1) / thread_num + 1;
    for one_batch in test_data.chunks(batch_size) {
        test_data_vec.push(one_batch.to_vec())
    }

    test_data.clear();
    test_data.shrink_to_fit();
    let t2 = PreciseTime::now();
    println!("split data: {}", t1.to(t2));

    // Create `thread_num` threads. Call gbdt::predict in parallel
    let t1 = PreciseTime::now();
    let gbdt_arc = Arc::new(gbdt);
    for data in test_data_vec.into_iter() {
        let gbdt_clone = Arc::clone(&gbdt_arc);
        let handle = thread::spawn(move || gbdt_clone.predict(&data));
        handles.push(handle)
    }

    // collect results
    let mut preds = Vec::with_capacity(data_size);
    for handle in handles {
        preds.append(&mut handle.join().unwrap());
    }

    let t2 = PreciseTime::now();
    println!("predict data: {}", t1.to(t2));
    assert_eq!(preds.len(), data_size);
}
