use gbdt::input::{InputFormat, load};
use time::PreciseTime;
fn main() {
    let start = PreciseTime::now();
    let train_file = "/Users/icst/workspace/xgboost/dataset/higgs/train.csv";
    let mut fmt = InputFormat::csv_format();
    fmt.set_feature_size(29);
    let mut train_dv = load(train_file, fmt);
    let end = PreciseTime::now();
    println!("New input time: {}", start.to(end));
}
