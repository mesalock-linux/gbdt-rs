//! This module is implements the config for gradient boosting.
//! 

use crate::decision_tree::ValueType;

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Loss {
    SquaredError,
    LogLikelyhood,
    LAD,
}

impl Default for Loss {
    fn default() -> Self {
        Loss::SquaredError
    }
}

#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Config {
    pub feature_size: usize,
    pub max_depth: u32,
    pub iterations: usize,
    pub shrinkage: ValueType,
    pub feature_sample_ratio: f64,
    pub data_sample_ratio: f64,
    pub min_leaf_size: usize,
    pub loss: Loss,
    pub debug: bool,
    pub feature_cost: Vec<f64>,
    pub feature_tunning_enabled: bool,
    pub initial_guess_enabled: bool,
}

pub fn string2loss(s: &str) -> Loss {
    match s {
        "LogLikelyhood" => Loss::LogLikelyhood,
        "SquaredError" => Loss::SquaredError,
        "LAD" => Loss::LAD,
        _ => Loss::SquaredError,
    }
}

pub fn loss2string(l: &Loss) -> String {
    match l {
        Loss::LogLikelyhood => String::from("LogLikelyhood"),
        Loss::SquaredError => String::from("SquaredError"),
        Loss::LAD => String::from("LAD"),
    }
}

impl Config {
    pub fn new() -> Config {
        Config {
            feature_size: 0,
            max_depth: 0,
            iterations: 0,
            shrinkage: 1.0,
            feature_sample_ratio: 1.0,
            data_sample_ratio: 1.0,
            min_leaf_size: 0,
            loss: Loss::SquaredError,
            debug: false,
            feature_cost: Vec::new(),
            feature_tunning_enabled: false,
            initial_guess_enabled: false,
        }
    }

    pub fn set_feature_size(&mut self, n: usize) {
        self.feature_size = n;
    }

    pub fn set_max_depth(&mut self, n: u32) {
        self.max_depth = n;
    }

    pub fn set_iterations(&mut self, n: usize) {
        self.iterations = n;
    }

    pub fn set_feature_sample_ratio(&mut self, n: f64) {
        self.feature_sample_ratio = n;
    }

    pub fn set_data_sample_ratio(&mut self, n: f64) {
        self.data_sample_ratio = n;
    }

    pub fn set_min_leaf_size(&mut self, n: usize) {
        self.min_leaf_size = n;
    }

    pub fn set_loss(&mut self, l: &str) {
        self.loss = string2loss(&l);
    }

    pub fn set_debug(&mut self, option: bool) {
        self.debug = option;
    }

    pub fn enable_feature_tunning(&mut self, option: bool) {
        self.feature_tunning_enabled = option;
    }

    pub fn enabled_initial_guess(&mut self, option: bool) {
        self.initial_guess_enabled = option;
    }

    pub fn to_string(&self) -> String {
        let mut s = String::from("");
        s.push_str(&format!(
            "number of features = {}\n",
            self.feature_size
        ));
        s.push_str(&format!("min leaf size = {}\n", self.min_leaf_size));
        s.push_str(&format!("maximum depth = {}\n", self.max_depth));
        s.push_str(&format!("iterations = {}\n", self.iterations));
        s.push_str(&format!("shrinkage = {}\n", self.shrinkage));
        s.push_str(&format!(
            "feature sample ratio = {}\n",
            self.feature_sample_ratio
        ));
        s.push_str(&format!("data sample ratio = {}\n", self.data_sample_ratio));
        s.push_str(&format!("debug enabled = {}\n", self.debug));
        s.push_str(&format!("loss type = {}\n", loss2string(&self.loss)));
        s.push_str(&format!(
            "feature tuning enabled = {}\n",
            self.feature_tunning_enabled
        ));
        s.push_str(&format!(
            "initial guess enabled = {}\n",
            self.initial_guess_enabled
        ));
        s
    }
}
