#[derive(Debug, PartialEq, Clone)]
pub enum LOSS {
    SquaredError,
    LogLikelyhood,
    LAD,
    UnknownLoss,
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
        "LogLikelyhood" => LOSS::LogLikelyhood,
        "SquaredError" => LOSS::SquaredError,
        "LAD" => LOSS::LAD,
        _ => LOSS::UnknownLoss,
    }
}

pub fn loss2string(l: &LOSS) -> String {
    match l {
        LOSS::LogLikelyhood => String::from("LogLikelyhood"),
        LOSS::SquaredError => String::from("SquaredError"),
        LOSS::LAD => String::from("LAD"),
        LOSS::UnknownLoss => String::from("UnknownLoss"),
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
            loss: LOSS::SquaredError,
            debug: false,
            feature_cost: Vec::new(),
            enable_feature_tunning: false,
            enable_initial_guess: false,
        }
    }
    pub fn to_string(&self) -> String {
        let mut s = String::from("");
        s.push_str(&format!(
            "number of features = {}\n",
            self.number_of_feature
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
            self.enable_feature_tunning
        ));
        s.push_str(&format!(
            "initial guess enabled = {}\n",
            self.enable_initial_guess
        ));
        s
    }
}
