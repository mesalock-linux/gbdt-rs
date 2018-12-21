// use continous variables for decision tree
type ValueType = f32;


pub struct Data {
    pub feature: Vec<ValueType>,  
    pub target: ValueType,
    pub weight: ValueType,
    pub initial_guess: ValueType,
}


type DataVec = Vec<Data>;
type PredVec = Vec<ValueType>;

pub struct DecisionTree {}

impl DecisionTree {
    pub fn new() -> Self {
        DecisionTree {}
    }
    pub fn fit(&mut self, train_data: &DataVec) {
    }
    pub fn predict(&self, test_data: &DataVec) -> PredVec {
        Vec::new()
    }
}
