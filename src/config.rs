//! This module implements the config for gradient boosting.
//!
//! Following hyperparameters are supported:
//!
//! 1. feature_size: the size of features. Training data and test data should have
//!    the same feature size. (default = 1)
//!
//! 2. max_depth: the max depth of a single decision tree. The root node is considered
//!    to be in the layer 0. (default = 2)
//!
//! 3. iterations: the iterations for training, which is also the number of trees in the
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
//! 8. loss: the loss function type. SquaredError, LogLikelyhood and LAD are supported for training and inference.
//!    RegLinear, RegLogistic, BinaryLogistic, BinaryLogitraw, MultiSoftprob, MultiSoftmax, RankPairwise are supported for inference with xgboost's model.
//!    See [Loss](enum.Loss.html). (default = SquareError)
//!
//! 9. debug: whether the debug information should be outputed. (default = false)
//!
//! 10. initial_guess_enabled: whether initial guess for test data is enabled. (default = false)
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
//! // initial guess enabled = false
//! ```

#[cfg(all(feature = "mesalock_sgx", not(target_env = "sgx")))]
use std::prelude::v1::*;

use crate::decision_tree::ValueType;
use serde_derive::{Deserialize, Serialize};

/// This enum defines the loss type.
///
/// We support three loss types for training and inference:
///
/// 1. SquaredError for regression. The label and the predicted value will be a float number.
/// 2. LogLikelyhood for binary classification. The label value should be -1 or 1. The predicted value should be a float number between 0 and 1, which is the possibility of label 1.
/// 3. LAD for regression. The label and the predicted value will be a float number.
///
/// Note that `LogLikelyhood` only support binary classification.
///
/// We also suppot seven objectives from Xgboost for inference. See [xgboost](https://xgboost.readthedocs.io/en/latest/parameter.html)
/// 1. RegLinear ("reg:linear" in xgboost): linear regression.
/// 2. RegLogistic ("reg:logistic" in xgboost): logistic regression.
/// 3. BinaryLogistic ("binary:logistic" in xgboost): logistic regression for binary classification, output probability
/// 4. BinaryLogitraw ("binary:logitraw" in xgboost): logistic regression for binary classification, output score before logistic transformation
/// 5. MultiSoftprob ("multi:softprob" in xgboost):  multiclass classification. Call [gbdt::predict_multiclass](../gradient_boost/struct.GBDT.html#method.predict_multiclass) to get the predictions.
/// 6. MultiSoftmax ("multi:softmax" in xgboost): multiclass classification. Call [gbdt::predict_multiclass](../gradient_boost/struct.GBDT.html#method.predict_multiclass) to get the predictions.
/// 7. RankPairwise ("rank:pairwise" in xgboost): pairwise rank. See [xgboost's demo](https://github.com/dmlc/xgboost/tree/master/demo/rank)
#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
pub enum Loss {
    /// SquaredError ("SquaredError") for regression. The label and the predicted value will be a float number.
    SquaredError,

    /// LogLikelyhood ("LogLikelyhood") for binary classification. The label value should be -1 or 1. The predicted value should be a float number between 0 and 1, which is the possibility of label 1.
    LogLikelyhood,

    /// LAD ("LAD") for regression. The label and the predicted value will be a float number.
    LAD,

    /// RegLinear ("reg:linear") from Xgboost: linear regression.
    RegLinear,

    /// RegLogistic ("reg:logistic") from Xgboost: logistic regression.
    RegLogistic,

    /// BinaryLogistic ("binary:logistic") from Xgboost: logistic regression for binary classification, output probability
    BinaryLogistic,

    /// BinaryLogitraw ("binary:logitraw") from Xgboost: logistic regression for binary classification, output score before logistic transformation
    BinaryLogitraw,

    /// MultiSoftprob ("multi:softprob") from Xgboost:  multiclass classification. Call [gbdt::predict_multiclass](../gradient_boost/struct.GBDT.html#method.predict_multiclass) to get the predictions.
    MultiSoftprob,

    /// MultiSoftmax ("multi:softmax") from Xgboost: multiclass classification. Call [gbdt::predict_multiclass](../gradient_boost/struct.GBDT.html#method.predict_multiclass) to get the predictions.
    MultiSoftmax,

    /// RankPairwise ("rank:pairwise") from Xgboost: pairwise rank. See [xgboost's demo](https://github.com/dmlc/xgboost/tree/master/demo/rank)
    RankPairwise,
}

impl Default for Loss {
    /// SquaredError are used as default loss type.
    fn default() -> Self {
        Loss::SquaredError
    }
}

/// Converting [std::string::String](https://doc.rust-lang.org/std/string/struct.String.html) to [Loss](enum.Loss.html).
///
/// # Example
/// ```rust
/// use gbdt::config::{Loss, string2loss};
///
/// let loss = string2loss("SquaredError");
/// ```
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
        "rank:pairwise" => Loss::RankPairwise,
        _ => {
            println!("unsupported loss, set to default(SquaredError)");
            Loss::SquaredError
        }
    }
}

/// Converting [Loss](enum.Loss.html) to [std::string::String](https://doc.rust-lang.org/std/string/struct.String.html).
///
/// # Example
/// ```rust
/// use gbdt::config::{Loss, loss2string};
/// println!("{}", loss2string(&Loss::SquaredError));
/// ```
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
        Loss::RankPairwise => String::from("rank:pairwise"),
    }
}

/// The config for the gradient boosting algorithm.
#[derive(Default, Clone, Serialize, Deserialize)]
pub struct Config {
    /// The size of features. Training data and test data should have the same feature size. (default = 1)
    pub feature_size: usize,

    /// The max depth of a single decision tree. The root node is considered to be in the layer 0. (default = 2)
    pub max_depth: u32,

    /// The iterations to train, which is also the number of trees in the gradient boosting algorithm. (default = 2)
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

    /// Whether initial guess for test data is enabled. (default = false)
    pub initial_guess_enabled: bool,

    /// Training optimization level (default = 2).
    ///
    /// 0: least memory, slowest speed.
    ///
    /// 1: more memory usage, faster speed.
    ///
    /// 2: most memory usage, fastest speed.
    pub training_optimization_level: u8,
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
            initial_guess_enabled: false,
            training_optimization_level: 2,
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

    /// Set learning rate.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_shrinkage(1.0);
    /// ```
    pub fn set_shrinkage(&mut self, eta: ValueType) {
        self.shrinkage = eta;
    }

    /// Set training optimization level (default = 2).
    ///
    /// 0: least memory, slowest speed.
    ///
    /// 1: more memory usage, faster speed.
    ///
    /// 2: most memory usage, fastest speed.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// let mut cfg = Config::new();
    /// cfg.set_training_optimization_level(2);
    /// ```
    pub fn set_training_optimization_level(&mut self, level: u8) {
        let optimization_level = if level >= 3 { 2 } else { level };
        self.training_optimization_level = optimization_level;
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

    /// Set loss type: "SquaredError", "LogLikelyhood", "LAD", "reg:linear", "binary:logistic", "reg:logistic", "binary:logitraw", "multi:softprob",  "multi:softmax",  "rank:pairwise"
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::{Config, Loss, loss2string};
    /// let mut cfg = Config::new();
    /// cfg.set_loss("LAD");
    /// // Alternative way
    /// cfg.set_loss(&loss2string(&Loss::SquaredError));
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

    /// Set whether initial guess of test data is enabled.
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
            "initial guess enabled = {}\n",
            self.initial_guess_enabled
        ));
        s
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{loss2string, string2loss, Config, Loss};

    const STRINGLOSS: [(&'static str, Loss); 11] = [
        ("LogLikelyhood", Loss::LogLikelyhood),
        ("SquaredError", Loss::SquaredError),
        ("LAD", Loss::LAD),
        ("reg:linear", Loss::RegLinear),
        ("binary:logistic", Loss::BinaryLogistic),
        ("reg:logistic", Loss::RegLogistic),
        ("binary:logitraw", Loss::BinaryLogitraw),
        ("multi:softprob", Loss::MultiSoftprob),
        ("multi:softmax", Loss::MultiSoftmax),
        ("rank:pairwise", Loss::RankPairwise),
        ("unknown", Loss::SquaredError),
    ];

    #[test]
    fn doc_test_config_head() {
        let mut cfg = Config::new();
        cfg.set_feature_size(4);
        cfg.set_max_depth(3);
        cfg.set_iterations(3);
        cfg.set_loss("LAD");

        assert_eq!(cfg.feature_size, 4);
        assert_eq!(cfg.max_depth, 3);
        assert_eq!(cfg.iterations, 3);
        assert_eq!(cfg.loss, Loss::LAD);
    }

    #[test]
    fn doc_test_string2loss() {
        for (s, l) in &STRINGLOSS {
            assert_eq!(string2loss(s), *l);
        }
    }

    #[test]
    fn doc_test_loss2string() {
        for (s, l) in &STRINGLOSS[..10] {
            assert_eq!(loss2string(l), *s);
        }
    }

    #[test]
    fn doc_test_config_new() {
        let cfg = Config::new();
        assert_eq!(cfg.feature_size, 1);
        assert_eq!(cfg.max_depth, 2);
        assert_eq!(cfg.iterations, 2);
        assert_eq!(cfg.shrinkage, 1.0);
        assert_eq!(cfg.feature_sample_ratio, 1.0);
        assert_eq!(cfg.data_sample_ratio, 1.0);
        assert_eq!(cfg.min_leaf_size, 1);
        assert_eq!(cfg.loss, Loss::SquaredError);
        assert_eq!(cfg.debug, false);
        assert_eq!(cfg.initial_guess_enabled, false);
        assert_eq!(cfg.training_optimization_level, 2);
    }

    #[test]
    fn doc_test_set_feature_size() {
        let mut cfg = Config::new();
        cfg.set_feature_size(10);
        assert_eq!(cfg.feature_size, 10);
        cfg.set_feature_size(20);
        assert_eq!(cfg.feature_size, 20);
    }

    #[test]
    fn doc_test_set_shrinkage() {
        let mut cfg = Config::new();
        cfg.set_shrinkage(3.0);
        assert_eq!(cfg.shrinkage, 3.0);
        cfg.set_shrinkage(5.0);
        assert_eq!(cfg.shrinkage, 5.0);
    }

    #[test]
    fn doc_test_set_training_optimization_level() {
        let mut cfg = Config::new();
        cfg.set_training_optimization_level(0);
        assert_eq!(cfg.training_optimization_level, 0);
        cfg.set_training_optimization_level(1);
        assert_eq!(cfg.training_optimization_level, 1);
        cfg.set_training_optimization_level(2);
        assert_eq!(cfg.training_optimization_level, 2);
        cfg.set_training_optimization_level(3);
        assert_eq!(cfg.training_optimization_level, 2);
        cfg.set_training_optimization_level(100);
        assert_eq!(cfg.training_optimization_level, 2);
    }

    #[test]
    fn doc_test_set_iterations() {
        let mut cfg = Config::new();
        cfg.set_iterations(1);
        assert_eq!(cfg.iterations, 1);
        cfg.set_iterations(10);
        assert_eq!(cfg.iterations, 10);
        cfg.set_iterations(100);
        assert_eq!(cfg.iterations, 100);
    }

    #[test]
    fn doc_test_set_feature_sample_ratio() {
        let mut cfg = Config::new();
        cfg.set_feature_sample_ratio(1.0);
        assert_eq!(cfg.feature_sample_ratio, 1.0);
        cfg.set_feature_sample_ratio(0.9);
        assert_eq!(cfg.feature_sample_ratio, 0.9);
        cfg.set_feature_sample_ratio(1.8);
        assert_eq!(cfg.feature_sample_ratio, 1.8);
    }

    #[test]
    fn doc_test_set_data_sample_ratio() {
        let mut cfg = Config::new();
        cfg.set_data_sample_ratio(1.0);
        assert_eq!(cfg.data_sample_ratio, 1.0);
        cfg.set_data_sample_ratio(0.9);
        assert_eq!(cfg.data_sample_ratio, 0.9);
        cfg.set_data_sample_ratio(1.8);
        assert_eq!(cfg.data_sample_ratio, 1.8);
    }

    #[test]
    fn doc_test_min_leaf_size() {
        let mut cfg = Config::new();
        cfg.set_min_leaf_size(1);
        assert_eq!(cfg.min_leaf_size, 1);
        cfg.set_min_leaf_size(10);
        assert_eq!(cfg.min_leaf_size, 10);
        cfg.set_min_leaf_size(100);
        assert_eq!(cfg.min_leaf_size, 100);
    }

    #[test]
    fn doc_test_set_loss() {
        let mut cfg = Config::new();
        for (s, l) in &STRINGLOSS {
            cfg.set_loss(s);
            assert_eq!(cfg.loss, *l);
        }
    }

    #[test]
    fn doc_test_set_debug() {
        let mut cfg = Config::new();
        cfg.set_debug(true);
        assert_eq!(cfg.debug, true);
        cfg.set_debug(false);
        assert_eq!(cfg.debug, false);
    }

    #[test]
    fn doc_test_to_string() {
        let cfg = Config::new();
        assert_eq!(cfg.to_string(), "number of features = 1\nmin leaf size = 1\nmaximum depth = 2\niterations = 2\nshrinkage = 1\nfeature sample ratio = 1\ndata sample ratio = 1\ndebug enabled = false\nloss type = SquaredError\ninitial guess enabled = false\n");
    }
}
