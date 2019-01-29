//! This module implements the process of gradient boosting decision tree
//! algorithm. This module depends on the following module:
//!
//! 1. [gbdt::config::Config](../config/): [Config](../config/struct.Config.html) is needed to configure the gbdt algorithm.
//!
//! 2. [gbdt::decision_tree](../decision_tree/): [DecisionTree](../decision_tree/struct.DecisionTree.html) is used
//!    for training and predicting.
//!
//! 3. [rand](https://docs.rs/rand/0.6.1/rand/): This standard module is used to randomly select the data or
//!    features used in a single iteration of training if the
//!    [data_sample_ratio](../config/struct.Config.html#structfield.data_sample_ratio) or
//!    [feature_sample_ratio](../config/struct.Config.html#structfield.feature_sample_ratio) is less than 1.0 .
//!
//! # Example
//! ```rust
//! use gbdt::config::Config;
//! use gbdt::gradient_boost::GBDT;
//! use gbdt::decision_tree::{Data, DataVec};
//!
//! // set config for algorithm
//! let mut cfg = Config::new();
//! cfg.set_feature_size(3);
//! cfg.set_max_depth(2);
//! cfg.set_min_leaf_size(1);
//! cfg.set_loss("SquaredError");
//! cfg.set_iterations(2);
//!
//! // initialize GBDT algorithm
//! let mut gbdt = GBDT::new(&cfg);
//!
//! // setup training data
//! let data1 = Data {
//!     feature: vec![1.0, 2.0, 3.0],
//!     target: 2.0,
//!     weight: 1.0,
//!     label: 1.0,
//!     residual: 1.0,
//!     initial_guess: 1.0,
//! };
//! let data2 = Data {
//!     feature: vec![1.1, 2.1, 3.1],
//!     target: 1.0,
//!     weight: 1.0,
//!     label: 1.0,
//!     residual: 1.0,
//!     initial_guess: 1.0,
//! };
//! let data3 = Data {
//!     feature: vec![2.0, 2.0, 1.0],
//!     target: 0.5,
//!     weight: 1.0,
//!     label: 2.0,
//!     residual: 2.0,
//!     initial_guess: 2.0,
//! };
//! let data4 = Data {
//!     feature: vec![2.0, 2.3, 1.2],
//!     target: 3.0,
//!     weight: 1.0,
//!     label: 0.0,
//!     residual: 0.0,
//!     initial_guess: 1.0,
//! };
//!
//! let mut dv: DataVec = Vec::new();
//! dv.push(data1.clone());
//! dv.push(data2.clone());
//! dv.push(data3.clone());
//! dv.push(data4.clone());
//!
//! // train the decision trees.
//! gbdt.fit(&mut dv);
//!
//! // setup the test data
//!
//! let mut dv: DataVec = Vec::new();
//! dv.push(data1.clone());
//! dv.push(data2.clone());
//! dv.push(data3.clone());
//! dv.push(data4.clone());
//!
//! println!("{:?}", gbdt.predict(&dv));
//!
//! // output:
//! // [1.0, 1.0, 2.0, 0.0]
//! ```

use crate::config::{Config, Loss};
use crate::decision_tree::DecisionTree;
use crate::decision_tree::{DataVec, PredVec, ValueType, VALUE_TYPE_UNKNOWN, TrainingCache};
use crate::fitness::*;
use rand::prelude::SliceRandom;
use rand::rngs::StdRng;
use rand::SeedableRng;
extern crate time;
use time::PreciseTime;

/// The gradient boosting decision tree.
#[derive(Default, Serialize, Deserialize)]
pub struct GBDT {
    /// The config of gbdt. See [gbdt::config](../config/) for detail.
    conf: Config,
    /// The trained decision trees.
    trees: Vec<DecisionTree>,
    /// The bias estimated.
    bias: ValueType,
    /// The information gain for each feature.
    pub gain: Vec<ValueType>,
}

impl GBDT {
    /// Return a new gbdt with manually set config.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// use gbdt::gradient_boost::GBDT;
    ///
    /// // set config for algorithm
    /// let mut cfg = Config::new();
    /// cfg.set_feature_size(3);
    /// cfg.set_max_depth(2);
    /// cfg.set_min_leaf_size(1);
    /// cfg.set_loss("SquaredError");
    /// cfg.set_iterations(2);
    ///
    /// // initialize GBDT algorithm
    /// let mut gbdt = GBDT::new(&cfg);
    /// ```
    pub fn new(conf: &Config) -> GBDT {
        GBDT {
            conf: conf.clone(),
            trees: Vec::new(),
            bias: 0.0,
            gain: Vec::new(),
        }
    }

    /// Return true if the data in the given data vector are all valid. In other case
    /// returns false.
    ///
    /// We simply check whether the length of feature vector in each data
    /// equals to the specified feature size in config.
    pub fn check_valid_data(&self, dv: &DataVec) -> bool {
        for d in dv {
            if d.feature.len() != self.conf.feature_size {
                return false;
            }
        }
        true
    }

    /// If initial_guess_enabled is set false in gbdt config, this function will calculate
    /// bias for initial guess based on train data. Different methods will be used according
    /// to different loss type. This is a private method and should not be called manually.
    ///
    /// # Panic
    /// If  specified length is greater than the length of data vector, it will panic.
    ///
    /// If there is invalid data that will confuse the training process, it will panic.
    fn init(&mut self, len: usize, dv: &DataVec) {
        assert!(dv.len() >= len);

        if !self.check_valid_data(&dv) {
            panic!("There are invalid data in data vector, check your data please.");
        }

        if self.conf.initial_guess_enabled {
            return;
        }

        self.bias = match self.conf.loss {
            Loss::SquaredError => label_average(dv, len),
            Loss::LogLikelyhood => {
                let v: ValueType = label_average(dv, len);
                ((1.0 + v) / (1.0 - v)).ln() / 2.0
            }
            Loss::LAD => weighted_label_median(dv, len),
        }
    }

    /// Fit the train data.
    ///
    /// First, initialize and configure decision trees. Then train the data for certain
    /// iterations set by config.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// use gbdt::gradient_boost::GBDT;
    /// use gbdt::decision_tree::{Data, DataVec, PredVec, ValueType};
    ///
    /// // set config for algorithm
    /// let mut cfg = Config::new();
    /// cfg.set_feature_size(3);
    /// cfg.set_max_depth(2);
    /// cfg.set_min_leaf_size(1);
    /// cfg.set_loss("SquaredError");
    /// cfg.set_iterations(2);
    ///
    /// // initialize GBDT algorithm
    /// let mut gbdt = GBDT::new(&cfg);
    ///
    /// // setup training data
    /// let data1 = Data {
    ///     feature: vec![1.0, 2.0, 3.0],
    ///     target: 2.0,
    ///     weight: 1.0,
    ///     label: 1.0,
    ///     residual: 1.0,
    ///     initial_guess: 1.0,
    /// };
    /// let data2 = Data {
    ///     feature: vec![1.1, 2.1, 3.1],
    ///     target: 1.0,
    ///     weight: 1.0,
    ///     label: 1.0,
    ///     residual: 1.0,
    ///     initial_guess: 1.0,
    /// };
    /// let data3 = Data {
    ///     feature: vec![2.0, 2.0, 1.0],
    ///     target: 0.5,
    ///     weight: 1.0,
    ///     label: 2.0,
    ///     residual: 2.0,
    ///     initial_guess: 2.0,
    /// };
    /// let data4 = Data {
    ///     feature: vec![2.0, 2.3, 1.2],
    ///     target: 3.0,
    ///     weight: 1.0,
    ///     label: 0.0,
    ///     residual: 0.0,
    ///     initial_guess: 1.0,
    /// };
    ///
    /// let mut dv: DataVec = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    /// // train the decision trees.
    /// gbdt.fit(&mut dv);
    /// ```
    pub fn fit(&mut self, train_data: &mut DataVec) {
        self.trees = Vec::with_capacity(self.conf.iterations);
        for i in 0..self.conf.iterations {
            self.trees.push(DecisionTree::new());
            self.trees[i].set_feature_size(self.conf.feature_size);
            self.trees[i].set_max_depth(self.conf.max_depth);
            self.trees[i].set_min_leaf_size(self.conf.min_leaf_size);
            self.trees[i].set_loss(self.conf.loss.clone());
        }
        let nr_samples: usize = if self.conf.data_sample_ratio < 1.0 {
            ((train_data.len() as f64) * self.conf.data_sample_ratio) as usize
        } else {
            train_data.len()
        };

        self.init(train_data.len(), &train_data);

        //let t1 = PreciseTime::now();
        //let mut train_data_copy = train_data.to_vec();
        //let t2 = PreciseTime::now();
        //println!("copy {}", t1.to(t2));

        //let mut rng = thread_rng();
        let seed = rand_seed();
        let mut rng: StdRng = SeedableRng::from_seed(seed.clone());
        //let mut rng_clone: StdRng = SeedableRng::from_seed(seed.clone());
        let mut predicted_cache: PredVec = self.predict_n(train_data, 0, train_data.len());
        println!("predicted_cache {}", predicted_cache[0]);
        //let mut train_data_copy = train_data.to_vec();

        //let t1 = PreciseTime::now();
        let mut cache = TrainingCache::get_cache(self.conf.feature_size, &train_data);
        //let t2 = PreciseTime::now();
        //println!("cache {}", t1.to(t2));



        for i in 0..self.conf.iterations {
            let t1 = PreciseTime::now();
            let mut samples: Vec<usize> = (0..train_data.len()).collect();
            let (subset, remain)  = if nr_samples < train_data.len() {
                samples.shuffle(&mut rng);
                let (left, right) = samples.split_at(nr_samples);
                let mut left = left.to_vec();
                let mut right = right.to_vec();
                left.sort();
                right.sort();
                (left, right)
            } else {
                (samples, Vec::new())
            };

            if self.conf.loss == Loss::SquaredError {
                self.square_loss_process(train_data, nr_samples, &predicted_cache);
            } else if self.conf.loss == Loss::LogLikelyhood {
                self.log_loss_process(train_data, nr_samples, &predicted_cache);
            } else if self.conf.loss == Loss::LAD {
                self.lad_loss_process(train_data, nr_samples, &predicted_cache);
            }
            self.trees[i].fit_n(train_data, &subset, &mut cache);
            let train_preds = cache.get_preds();
            for index in subset.iter() {
                predicted_cache[*index] += train_preds[*index] * self.conf.shrinkage;
            }
            //self.trees[i].fit_n(&train_data_copy, nr_samples);
            let predicted_tmp= self.trees[i].predict_n(train_data, &remain);
            for index in remain.iter() {
                predicted_cache[*index] += predicted_tmp[*index] * self.conf.shrinkage;
            }

            let t2 = PreciseTime::now();
            println!("one iter {} {}", i, t1.to(t2));

        }
    }


    /*
    #[inline(always)]
    pub fn predict_nth(&self, test_data: &DataVec, iter: usize, n: usize) -> PredVec {
        assert!(test_data.len() >= n);
        assert!(iter < self.trees.len());
        self.trees[iter].predict_n(&test_data, n)
    }*/

    /// Predicting the first `n` data in data vector with the first `iters` trees.
    ///
    /// Note that the result will not be normalized no matter what loss type is used.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// use gbdt::gradient_boost::GBDT;
    /// use gbdt::decision_tree::{Data, DataVec, PredVec, ValueType};
    ///
    /// // set config for algorithm
    /// let mut cfg = Config::new();
    /// cfg.set_feature_size(3);
    /// cfg.set_max_depth(2);
    /// cfg.set_min_leaf_size(1);
    /// cfg.set_loss("SquaredError");
    /// cfg.set_iterations(2);
    ///
    /// // initialize GBDT algorithm
    /// let mut gbdt = GBDT::new(&cfg);
    ///
    /// // setup training data
    /// let data1 = Data {
    ///     feature: vec![1.0, 2.0, 3.0],
    ///     target: 2.0,
    ///     weight: 1.0,
    ///     label: 1.0,
    ///     residual: 1.0,
    ///     initial_guess: 1.0,
    /// };
    /// let data2 = Data {
    ///     feature: vec![1.1, 2.1, 3.1],
    ///     target: 1.0,
    ///     weight: 1.0,
    ///     label: 1.0,
    ///     residual: 1.0,
    ///     initial_guess: 1.0,
    /// };
    /// let data3 = Data {
    ///     feature: vec![2.0, 2.0, 1.0],
    ///     target: 0.5,
    ///     weight: 1.0,
    ///     label: 2.0,
    ///     residual: 2.0,
    ///     initial_guess: 2.0,
    /// };
    /// let data4 = Data {
    ///     feature: vec![2.0, 2.3, 1.2],
    ///     target: 3.0,
    ///     weight: 1.0,
    ///     label: 0.0,
    ///     residual: 0.0,
    ///     initial_guess: 1.0,
    /// };
    ///
    /// let mut dv: DataVec = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    /// // train the decision trees.
    /// gbdt.fit(&mut dv);
    ///
    /// // setup the test data
    ///
    /// let mut dv: DataVec = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    /// println!("{:?}", gbdt.predict_n(&dv, 2, dv.len()));
    /// ```
    ///
    /// # Panic
    /// If n is greater than the length of test data vector, it will panic.
    ///
    /// If the iterations is greater than the number of trees that have been trained, it will panic.
    pub fn predict_n(&self, test_data: &DataVec, iters: usize, n: usize) -> PredVec {
        assert!(iters <= self.trees.len());
        assert!(n <= test_data.len());

        if self.trees.is_empty() {
            return vec![VALUE_TYPE_UNKNOWN; test_data.len()];
        }

        let mut predicted: PredVec = if !self.conf.initial_guess_enabled {
            vec![self.bias; n]
        } else {
            test_data.iter().take(n).map(|x| x.initial_guess).collect()
        };

        let subset: Vec<usize> = (0..n).collect();
        for i in 0..iters {
            let v: PredVec = self.trees[i].predict_n(&test_data, &subset);
            for d in 0..n {
                predicted[d] += self.conf.shrinkage * v[d];
            }
        }
        predicted
    }

    /// Predicting the given data.
    ///
    /// Note that when using log likelyhood loss type, the predicted value will be
    /// normalized to 1.0 and -1.0 .
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// use gbdt::gradient_boost::GBDT;
    /// use gbdt::decision_tree::{Data, DataVec, PredVec, ValueType};
    ///
    /// // set config for algorithm
    /// let mut cfg = Config::new();
    /// cfg.set_feature_size(3);
    /// cfg.set_max_depth(2);
    /// cfg.set_min_leaf_size(1);
    /// cfg.set_loss("SquaredError");
    /// cfg.set_iterations(2);
    ///
    /// // initialize GBDT algorithm
    /// let mut gbdt = GBDT::new(&cfg);
    ///
    /// // setup training data
    /// let data1 = Data {
    ///     feature: vec![1.0, 2.0, 3.0],
    ///     target: 2.0,
    ///     weight: 1.0,
    ///     label: 1.0,
    ///     residual: 1.0,
    ///     initial_guess: 1.0,
    /// };
    /// let data2 = Data {
    ///     feature: vec![1.1, 2.1, 3.1],
    ///     target: 1.0,
    ///     weight: 1.0,
    ///     label: 1.0,
    ///     residual: 1.0,
    ///     initial_guess: 1.0,
    /// };
    /// let data3 = Data {
    ///     feature: vec![2.0, 2.0, 1.0],
    ///     target: 0.5,
    ///     weight: 1.0,
    ///     label: 2.0,
    ///     residual: 2.0,
    ///     initial_guess: 2.0,
    /// };
    /// let data4 = Data {
    ///     feature: vec![2.0, 2.3, 1.2],
    ///     target: 3.0,
    ///     weight: 1.0,
    ///     label: 0.0,
    ///     residual: 0.0,
    ///     initial_guess: 1.0,
    /// };
    ///
    /// let mut dv: DataVec = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    /// // train the decision trees.
    /// gbdt.fit(&mut dv);
    ///
    /// // setup the test data
    ///
    /// let mut dv: DataVec = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    /// println!("{:?}", gbdt.predict(&dv));
    /// ```
    ///
    /// # Panic
    /// If the training process is not completed, thus, the number of trees that have been
    /// is less than the iteration configuration in `self.conf`, it will panic.
    pub fn predict(&self, test_data: &DataVec) -> PredVec {
        assert_eq!(self.conf.iterations, self.trees.len());
        let predicted = self.predict_n(test_data, self.conf.iterations, test_data.len());

        if self.conf.loss == Loss::LogLikelyhood {
            predicted
                .iter()
                .map(|x| {
                    if (1.0 / (1.0 + ((-2.0 * x).exp()))) >= 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                })
                .collect()
        } else {
            predicted
        }
    }

    /// Predicting the given data.
    ///
    /// # Example
    /// ```rust
    /// use gbdt::config::Config;
    /// use gbdt::gradient_boost::GBDT;
    /// use gbdt::decision_tree::{Data, DataVec, PredVec, ValueType};
    ///
    /// // set config for algorithm
    /// let mut cfg = Config::new();
    /// cfg.set_feature_size(3);
    /// cfg.set_max_depth(2);
    /// cfg.set_min_leaf_size(1);
    /// cfg.set_loss("SquaredError");
    /// cfg.set_iterations(2);
    ///
    /// // initialize GBDT algorithm
    /// let mut gbdt = GBDT::new(&cfg);
    ///
    /// // setup training data
    /// let data1 = Data {
    ///     feature: vec![1.0, 2.0, 3.0],
    ///     target: 2.0,
    ///     weight: 1.0,
    ///     label: 1.0,
    ///     residual: 1.0,
    ///     initial_guess: 1.0,
    /// };
    /// let data2 = Data {
    ///     feature: vec![1.1, 2.1, 3.1],
    ///     target: 1.0,
    ///     weight: 1.0,
    ///     label: 1.0,
    ///     residual: 1.0,
    ///     initial_guess: 1.0,
    /// };
    /// let data3 = Data {
    ///     feature: vec![2.0, 2.0, 1.0],
    ///     target: 0.5,
    ///     weight: 1.0,
    ///     label: 2.0,
    ///     residual: 2.0,
    ///     initial_guess: 2.0,
    /// };
    /// let data4 = Data {
    ///     feature: vec![2.0, 2.3, 1.2],
    ///     target: 3.0,
    ///     weight: 1.0,
    ///     label: 0.0,
    ///     residual: 0.0,
    ///     initial_guess: 1.0,
    /// };
    ///
    /// let mut dv: DataVec = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    /// // train the decision trees.
    /// gbdt.fit(&mut dv);
    ///
    /// // print the tree.
    /// gbdt.print_trees();
    pub fn print_trees(&self) {
        for i in 0..self.trees.len() {
            self.trees[i].print();
        }
    }

    /// This is the process to calculate the residual as the target in next iteration
    /// using squared error loss function. This is a private method that should not be
    /// called manually.
    fn square_loss_process(&self, dv: &mut DataVec, samples: usize, predicted: &PredVec) {
        // let predicted: PredVec = self.predict_n(&dv, iters, samples);
        for i in 0..samples {
            dv[i].target = dv[i].label - predicted[i];
        }
        if self.conf.debug {
            println!("RMSE = {}", RMSE(&dv, &predicted, samples));
        }
    }

    /// This is the process to calculate the residual as the target in next iteration
    /// using negative binomial log-likehood loss function. This is a private method that should not be
    /// called manually.
    fn log_loss_process(&self, dv: &mut DataVec, samples: usize, predicted: &PredVec) {
        // let predicted: PredVec = self.predict_n(&dv, iters, samples);
        for i in 0..samples {
            dv[i].target = logit_loss_gradient(dv[i].label, predicted[i]);
        }
    }

    /// This is the process to calculate the residual as the target in next iteration
    /// using LAD loss function. This is a private method that should not be
    /// called manually.
    fn lad_loss_process(&self, dv: &mut DataVec, samples: usize, predicted: &PredVec) {
        // let predicted: PredVec = self.predict_n(&dv, iters, samples);
        for i in 0..samples {
            dv[i].residual = dv[i].label - predicted[i];
            dv[i].target = if dv[i].residual >= 0.0 { 1.0 } else { -1.0 };
        }
        if self.conf.debug {
            println!("MAE {}", MAE(&dv, &predicted, samples));
        }
    }
}
