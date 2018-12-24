// use continous variables for decision tree
pub type ValueType = f64;

#[derive(Clone)]
pub struct Data {
    pub feature: Vec<ValueType>,  
    pub target: ValueType,
    pub weight: ValueType,
    pub label: ValueType,
    pub residual: ValueType,
    pub initial_guess: ValueType,
}


pub type DataVec = Vec<Data>;
pub type PredVec = Vec<ValueType>;

pub const VALUE_TYPE_MAX: f64 = std::f64::MAX;
pub const VALUE_TYPE_MIN: f64 = std::f64::MIN;
pub const VALUE_TYPE_UNKNOWN: f64 = VALUE_TYPE_MIN;

pub struct DecisionTree {}

impl DecisionTree {
    pub fn new() -> Self {
        DecisionTree {}
    }
    pub fn fit_n(&mut self, train_data: &DataVec, n: usize) {
    }
    pub fn fit(&mut self, train_data: &DataVec) {
    }
    pub fn predict_n(&self, test_data: &DataVec, n: usize) -> PredVec {
        vec![0.5; n]
    }
    pub fn predict(&self, test_data: &DataVec) -> PredVec {
        vec![0.5; test_data.len()]
    }
}
