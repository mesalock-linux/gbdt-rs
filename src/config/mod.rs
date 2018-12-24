
#[derive(Debug)]
#[derive(PartialEq)]
#[derive(Clone)]
#[allow(non_camel_case_types)]
pub enum LOSS {
    SQUARED_ERROR,
    LOG_LIKEHOOD,
    LAD,
    UNKNOWN_LOSS,
}

#[derive(Clone)]
pub struct Config {
    pub number_of_feature: usize,
    pub max_depth: u32,
    pub iterations: usize,
    pub shrinkage: f64,
    pub feature_sample_ratio: f64,
    pub data_sample_ratio: f64,
    pub min_leaf_size: u32,
    pub loss: LOSS,
    pub debug: bool,
    pub feature_cost: Vec<f64>,
    pub enable_feature_tunning: bool,
    pub enable_initial_guess: bool,
}

pub fn string2loss(s: &str) -> LOSS {
    match s {
        "LOG_LIKEHOOD" => LOSS::LOG_LIKEHOOD,
        "SQUARED_ERROR" => LOSS::SQUARED_ERROR,
        "LAD" => LOSS::LAD,
        _ => LOSS::UNKNOWN_LOSS,
    }
}

pub fn loss2string(l: LOSS) -> String {
    match l {
        LOSS::LOG_LIKEHOOD => String::from("LOG_LIKEHOOD"),
        LOSS::SQUARED_ERROR => String::from("SQUARED_ERROR"),
        LOSS::LAD => String::from("LAD"),
        LOSS::UNKNOWN_LOSS => String::from("UNKNOWN_LOSS"),
    }
}

impl Config {
    pub fn empty_config() -> Config {
        Config {
            number_of_feature: 0,
            max_depth: 1,
            iterations: 1,
            shrinkage: 1.0,
            feature_sample_ratio: 1.0,
            data_sample_ratio: 1.0,
            min_leaf_size: 0,
            loss: LOSS::SQUARED_ERROR,
            debug: false,
            feature_cost: Vec::new(),
            enable_feature_tunning: false,
            enable_initial_guess: false,
        }
    }
    pub fn to_string(self) -> String {
        let mut s = String::from("");
        s.push_str(&format!("number of features = {}\n", self.number_of_feature));
        s.push_str(&format!("min leaf size = {}\n", self.min_leaf_size));
        s.push_str(&format!("maximum depth = {}\n", self.max_depth));
        s.push_str(&format!("iterations = {}\n", self.iterations));
        s.push_str(&format!("shrinkage = {}\n", self.shrinkage));
        s.push_str(&format!("feature sample ratio = {}\n", self.feature_sample_ratio));
        s.push_str(&format!("data sample ratio = {}\n", self.data_sample_ratio));
        s.push_str(&format!("debug enabled = {}\n", self.debug));
        s.push_str(&format!("loss type = {}\n", loss2string(self.loss)));
        s.push_str(&format!("feature tuning enabled = {}\n", self.enable_feature_tunning));
        s.push_str(&format!("initial guess enabled = {}\n", self.enable_initial_guess));
        s
    }
}