//! This module is implements the config for gradient boosting.
//!
//! Following hyperparameters are supported:
//!
//! 1. feature_size: the size of features. Training data and test data should have
//!    the same feature size. (default = 1)
//!
//! 2. max_depth: the max depth of a single decision tree. The root node is considered
//!    to be in the layer 0. (default = 2)
//!
//! 3. iterations: the iterations to train, which is also the number of trees in the
//!    gradient boosting algorithm. (default = 2)
//!
//! 4. shrinkage: the learning rate parameter of the gradient boosting algorithm.
//!    (default = 1.0)
//!
//! 5. feature_sample_raio: portion of features to be splited. When spliting a node, a subset of
//!    the features (feature_size * feature_sample_ratio) will be randomly selected to calculate
//!    impurity. (default = 1.0)
//!
//! 6. data_sample_ratio: portion of data used to train in a single iteration. Data will
//!    be randomly selected for the training. (default = 1.0)
//!
//! 7. min_leaf_size: the minimum number of samples required to be at a leaf node during training.
//!    (default = 1)
//!
//! 8. loss: the loss function type. SquaredError, LogLikelyhood and LAD are supported. See
//!    [Loss](enum.Loss.html). (default = SquareError)
//!
//! 9. debug: whether the debug information should be outputed. (default = false)
//!
//! 10. feature_cost: the parameter to tune the model. Used if feature_tunning_enabled is set true.
//!     (default = Vec::new())
//!
//! 11. feature_tunning_enabled: whether feature tuning is enabled. When set true,
//!     `feature_costs' is used to tune the model. (default = false)
//!
//! 12. initial_guess_enabled: whether initial guess for test data is enabled. (default = false)
//!
//!
//! # Example
//! ```rust
//! use gbdt::config::Config;
//! let mut cfg = Config::new();
//! cfg.set_feature_size(4);
//! cfg.set_max_depth(3);
//! cfg.set_iterations(3);
//! cfg.set_loss("LAD");
//! println!("{}", cfg.to_string());
//!
//! // output
//! // number of features = 4
//! // min leaf size = 1
//! // maximum depth = 3
//! // iterations = 3
//! // shrinkage = 1
//! // feature sample ratio = 1
//! // data sample ratio = 1
//! // debug enabled = false
//! // loss type = LAD
//! // feature tuning enabled = false
//! // initial guess enabled = false
//! ```

use crate::decision_tree::ValueType;

/// This enum defines the loss type.
///
/// We support three loss types:
///
/// 1. SquaredError
/// 2. LogLikelyhood
/// 3. LAD
///
/// Note that `LogLikelyhood` only support two class classification.
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Loss {
    /// Squared Error loss type.
    SquaredError,

    /// Negative binomial log-likehood loss type.
    LogLikelyhood,

    /// LAD loss type
    LAD,

    /// XGBOOST
    RegLinear,

    RegLogistic,

    BinaryLogistic,

    BinaryLogitraw,

    MultiSoftprob,

    MultiSoftmax,
}

impl Default for Loss {
    /// SquaredError are used as default loss type.
    fn default() -> Self {
        Loss::SquaredError
    }
}

/// The config for the gradient boosting algorithm.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Config {
    /// The size of features. Training data and test data should have the same feature size. (default = 1)
    pub feature_size: usize,

    /// The max depth of a single decision tree. The root node is considered to be in the layer 0. (default = 2)
    pub max_depth: u32,

    /// The iterations to train, which is also the number of trees in thegradient boosting algorithm.
    pub iterations: usize,

    /// The learning rate parameter of the gradient boosting algorithm.(default = 1.0)
    pub shrinkage: ValueType,

    /// Portion of features to be splited. (default = 1.0)
    pub feature_sample_ratio: f64,

    /// Portion of data to be splited. (default = 1.0)
    pub data_sample_ratio: f64,

    /// The minimum number of samples required to be at a leaf node during training. (default = 1.0)
    pub min_leaf_size: usize,

    /// The loss function type. (default = SquareError)
    pub loss: Loss,

    /// Whether the debug information should be outputed. (default = false)
    pub debug: bool,

    /// The parameter to tune the model. (default = vec![])
    pub feature_cost: Vec<f64>,

    /// Whether feature tuning is enabled. (default = false)
    pub feature_tunning_enabled: bool,

    /// Whether initial guess for test data is enabled. (default = false)
    pub initial_guess_enabled: bool,
}

/// Converting [std::string::String](https://doc.rust-lang.org/std/string/struct.String.html) to [Loss](enum.Loss.html).
pub fn string2loss(s: &str) -> Loss {
    match s {
        "LogLikelyhood" => Loss::LogLikelyhood,
        "SquaredError" => Loss::SquaredError,
        "LAD" => Loss::LAD,
        "reg:linear" => Loss::RegLinear,
        "binary:logistic" => Loss::BinaryLogistic,
        "reg:logistic" => Loss::RegLogistic,
        "binary:logitraw" => Loss::BinaryLogitraw,
        "multi:softprob" => Loss::MultiSoftprob,
        "multi:softmax" => Loss::MultiSoftmax,
        _ => {
            println!("unsupported loss, set to default(SquaredError)");
            Loss::SquaredError
        }
    }
}

/// Converting [Loss](enum.Loss.html) to [std::string::String](https://doc.rust-lang.org/std/string/struct.String.html).
pub fn loss2string(l: &Loss) -> String {
    match l {
        Loss::LogLikelyhood => String::from("LogLikelyhood"),
        Loss::SquaredError => String::from("SquaredError"),
        Loss::LAD => String::from("LAD"),
        Loss::RegLinear => String::from("reg:linear"),
        Loss::BinaryLogistic => String::from("binary:logistic"),
        Loss::RegLogistic => String::from("reg:logistic"),
        Loss::BinaryLogitraw => String::from("binary:logitraw"),
        Loss::MultiSoftprob => String::from("multi:softprob"),
        Loss::MultiSoftmax => String::from("multi:softmax"),
    }
}

impl Config {
    /// Return a new config with default settings.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// ```
    pub fn new() -> Config {
        Config {
            feature_size: 1,
            max_depth: 2,
            iterations: 2,
            shrinkage: 1.0,
            feature_sample_ratio: 1.0,
            data_sample_ratio: 1.0,
            min_leaf_size: 1,
            loss: Loss::SquaredError,
            debug: false,
            feature_cost: Vec::new(),
            feature_tunning_enabled: false,
            initial_guess_enabled: false,
        }
    }

    /// Set feature size.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_feature_size(10);
    /// ```
    pub fn set_feature_size(&mut self, n: usize) {
        self.feature_size = n;
    }

    /// Set max depth of the tree.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_max_depth(5);
    /// ```
    pub fn set_max_depth(&mut self, n: u32) {
        self.max_depth = n;
    }

    /// Set iterations of the algorithm.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_iterations(5);
    /// ```
    pub fn set_iterations(&mut self, n: usize) {
        self.iterations = n;
    }

    /// Set feature sample ratio.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_feature_sample_ratio(0.9);
    /// ```
    pub fn set_feature_sample_ratio(&mut self, n: f64) {
        self.feature_sample_ratio = n;
    }

    /// Set data sample ratio.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_data_sample_ratio(0.9);
    /// ```
    pub fn set_data_sample_ratio(&mut self, n: f64) {
        self.data_sample_ratio = n;
    }

    /// Set minimal leaf size.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_min_leaf_size(3);
    /// ```
    pub fn set_min_leaf_size(&mut self, n: usize) {
        self.min_leaf_size = n;
    }

    /// Set loss type.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_loss("LAD");
    /// ```
    pub fn set_loss(&mut self, l: &str) {
        self.loss = string2loss(&l);
    }

    /// Set debug mode.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_debug(true);
    /// ```
    pub fn set_debug(&mut self, option: bool) {
        self.debug = option;
    }

    /// Set whther feature tunning is enabled.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.enable_feature_tunning(true);
    /// ```
    pub fn enable_feature_tunning(&mut self, option: bool) {
        self.feature_tunning_enabled = option;
    }

    /// Set whther initial guess of test data is enabled.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.enabled_initial_guess(false);
    /// ```
    pub fn enabled_initial_guess(&mut self, option: bool) {
        self.initial_guess_enabled = option;
    }

    /// Dump the config to string for presentation.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// println!("{}", cfg.to_string());
    /// ```
    pub fn to_string(&self) -> String {
        let mut s = String::from("");
        s.push_str(&format!("number of features = {}\n", self.feature_size));
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
