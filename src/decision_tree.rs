//! This module implements a decision tree from the simple binary tree [gbdt::binary_tree].
//!
//! In the training process, the nodes are splited according `impurity`.
//!
//! The decision tree accepts features represented by f64. Unknown features are not supported yet.
//!
//! Following hyperparameters are supported:
//!
//! 1. feature_size: the size of feautures. Training data and test data should have same
//!    feature_size. (default = 1)
//!
//! 2. max_depth: the max depth of the decision tree. The root node is considered to be in the layer
//!    0. (default = 2)
//!
//! 3. min_leaf_size: the minimum number of samples required to be at a leaf node during training.
//!    (default = 1)
//!
//! 4. loss: the loss function type. SquaredError, LogLikelyhood and LAD are supported. See
//!    [config::Loss]. (default = SquareError).
//!
//! 5. feature_sample_ratio: portion of features to be splited. When spliting a node, a subset of
//!    the features (feature_size * feature_sample_ratio) will be randomly selected to calculate
//!    impurity. (default = 1.0)
//!
//! [gbdt::binary_tree]: ../binary_tree/index.html
//!
//! [config::Loss]: ../config/enum.Loss.html
//!
//! # Example
//! ```
//! use gbdt::config::Loss;
//! use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
//! // set up training data
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
//! weight: 1.0,
//! label: 0.0,
//! residual: 0.0,
//! initial_guess: 1.0,
//! };
//!
//! let mut dv = Vec::new();
//! dv.push(data1.clone());
//! dv.push(data2.clone());
//! dv.push(data3.clone());
//! dv.push(data4.clone());
//!
//!
//! // train a decision tree
//! let mut tree = DecisionTree::new();
//! tree.set_feature_size(3);
//! tree.set_max_depth(2);
//! tree.set_min_leaf_size(1);
//! tree.set_loss(Loss::SquaredError);
//! let mut cache = TrainingCache::get_cache(3, &dv, 3);
//! tree.fit(&dv, &mut cache);
//!
//!
//! // set up the test data
//! let mut dv = Vec::new();
//! dv.push(data1.clone());
//! dv.push(data2.clone());
//! dv.push(data3.clone());
//! dv.push(data4.clone());
//!
//!
//! // inference the test data with the decision tree
//! println!("{:?}", tree.predict(&dv));
//!
//!
//! // output:
//! // [2.0, 0.75, 0.75, 3.0]
//! ```

use crate::binary_tree::BinaryTree;
use crate::binary_tree::BinaryTreeNode;
use crate::binary_tree::TreeIndex;
use crate::config::Loss;
use crate::fitness::almost_equal;
use std::error::Error;

use rand::prelude::SliceRandom;
use rand::thread_rng;

extern crate serde_json;

// For now we only support std::$t using this macro.
// We will generalize ValueType in future.
macro_rules! def_value_type {
    ($t: tt) => {
        pub type ValueType = $t;
        pub const VALUE_TYPE_MAX: ValueType = std::$t::MAX;
        pub const VALUE_TYPE_MIN: ValueType = std::$t::MIN;
        pub const VALUE_TYPE_UNKNOWN: ValueType = VALUE_TYPE_MIN;
    };
}

// use continous variables for decision tree
def_value_type!(f32);

struct ImpurityCache {
    sum_s: f64,
    sum_ss: f64,
    sum_c: f64,
    cached: bool,
    bool_vec: Vec<bool>,
    sample_size: usize,
}

impl ImpurityCache {
    fn new(sample_size: usize, train_data: &[usize]) -> Self {
        let mut bool_vec: Vec<bool> = vec![false; sample_size];
        for index in train_data.iter() {
            bool_vec[*index] = true;
        }
        ImpurityCache {
            sum_s: 0.0,
            sum_ss: 0.0,
            sum_c: 0.0,
            cached: false,
            bool_vec,
            sample_size,
        }
    }
}
struct CacheValue {
    s: ValueType,
    ss: ValueType,
    c: ValueType,
}
pub struct TrainingCache {
    ordered_features: Vec<Vec<(usize, ValueType)>>,
    ordered_residual: Vec<(usize, ValueType)>,
    cache_value: Vec<CacheValue>, //s, ss, c
    //ss: Vec<ValueType>,
    //s: Vec<ValueType>,
    //c: Vec<ValueType>,
    //target: Vec<ValueType>,
    logit_c: Vec<ValueType>,
    sample_size: usize,
    feature_size: usize,
    preds: Vec<ValueType>,
    cache_level: u8,
}

impl TrainingCache {
    pub fn get_cache(feature_size: usize, data: &DataVec, cache_level: u8) -> Self {
        let level = if cache_level >= 3 { 2 } else { cache_level };
        let sample_size = data.len();
        let logit_c = vec![0.0; data.len()];
        let preds = vec![VALUE_TYPE_UNKNOWN; sample_size];

        let mut cache_value = Vec::with_capacity(data.len());
        for elem in data {
            let item = CacheValue {
                s: 0.0,
                ss: 0.0,
                c: elem.weight,
            };
            cache_value.push(item);
        }

        let ordered_features: Vec<Vec<(usize, ValueType)>> = if (level == 0) || (level == 2) {
            TrainingCache::cache_features(data, feature_size)
        } else {
            Vec::new()
        };
        let ordered_residual: Vec<(usize, ValueType)> = Vec::new();

        TrainingCache {
            ordered_features,
            ordered_residual,
            cache_value,
            //target,
            logit_c,
            sample_size,
            feature_size,
            preds,
            cache_level: level,
        }

        //let mut target = Vec::with_capacity(data.len());
    }
    pub fn get_preds(&self) -> Vec<ValueType> {
        self.preds.to_vec()
    }

    pub fn init_one_iteration(&mut self, whole_data: &[Data], loss: &Loss) {
        for (index, data) in whole_data.iter().enumerate() {
            let target = data.target;
            let weight = data.weight;
            let s = target * weight;
            self.cache_value[index].s = s;
            self.cache_value[index].ss = target * s;
            if let Loss::LogLikelyhood = loss {
                let y = target.abs();
                let c = y * (2.0 - y) * weight;
                self.logit_c[index] = c;
            }
        }
        if let Loss::LAD = loss {
            self.ordered_residual = TrainingCache::cache_residual(whole_data);
        }
    }

    fn cache_features(whole_data: &[Data], feature_size: usize) -> Vec<Vec<(usize, ValueType)>> {
        let mut ordered_features = Vec::with_capacity(feature_size);
        for _index in 0..feature_size {
            let nv: Vec<(usize, ValueType)> = Vec::with_capacity(whole_data.len());
            ordered_features.push(nv);
        }
        for (i, item) in whole_data.iter().enumerate() {
            for index in 0..feature_size {
                ordered_features[index].push((i, item.feature[index]));
            }
        }
        for index in 0..feature_size {
            ordered_features[index].sort_unstable_by(|a, b| {
                let v1 = a.1;
                let v2 = b.1;
                v1.partial_cmp(&v2).unwrap()
            });
        }
        ordered_features
    }

    fn cache_residual(whole_data: &[Data]) -> Vec<(usize, ValueType)> {
        let mut ordered_residual = Vec::with_capacity(whole_data.len());
        for (index, elem) in whole_data.iter().enumerate() {
            ordered_residual.push((index, elem.residual));
        }
        ordered_residual.sort_unstable_by(|a, b| {
            let v1: ValueType = a.1;
            let v2: ValueType = b.1;
            v1.partial_cmp(&v2).unwrap()
        });
        ordered_residual
    }

    fn sort_with_bool_vec(
        &self,
        feature_index: usize,
        is_residual: bool,
        to_sort: &[bool],
        to_sort_size: usize,
        sub_cache: &SubCache,
    ) -> Vec<(usize, ValueType)> {
        let whole_data_sorted_index = if is_residual {
            if (self.cache_level == 0) || sub_cache.lazy {
                &self.ordered_residual
            } else {
                &sub_cache.ordered_residual
            }
        } else {
            if (self.cache_level == 0) || sub_cache.lazy {
                &self.ordered_features[feature_index]
            } else {
                &sub_cache.ordered_features[feature_index]
            }
        };
        let mut ret = Vec::with_capacity(to_sort_size);
        for item in whole_data_sorted_index.iter() {
            let (index, value) = *item;
            if to_sort[index] {
                ret.push((index, value));
            }
        }
        ret
    }
    fn sort_with_cache(
        &self,
        feature_index: usize,
        is_residual: bool,
        to_sort: &[usize],
        sub_cache: &SubCache,
    ) -> Vec<(usize, ValueType)> {
        let whole_data_sorted_index = if is_residual {
            &self.ordered_residual
        } else {
            &self.ordered_features[feature_index]
        };
        let mut index_exists: Vec<bool> = vec![false; whole_data_sorted_index.len()];
        for index in to_sort.iter() {
            index_exists[*index] = true;
        }
        self.sort_with_bool_vec(
            feature_index,
            is_residual,
            &index_exists,
            to_sort.len(),
            sub_cache,
        )
    }
}

struct SubCache {
    ordered_features: Vec<Vec<(usize, ValueType)>>,
    ordered_residual: Vec<(usize, ValueType)>,
    lazy: bool,
}

impl SubCache {
    fn get_cache_from_training_cache(cache: &TrainingCache, data: &[Data], loss: &Loss) -> Self {
        let level = cache.cache_level;
        if level == 2 {
            return SubCache {
                ordered_features: Vec::new(),
                ordered_residual: Vec::new(),
                lazy: true,
            };
        }

        let ordered_features = if level == 0 {
            Vec::new()
        } else if level == 1 {
            TrainingCache::cache_features(data, cache.feature_size)
        } else {
            let mut ordered_features: Vec<Vec<(usize, ValueType)>> =
                Vec::with_capacity(cache.feature_size);
            for index in 0..cache.feature_size {
                ordered_features.push(cache.ordered_features[index].to_vec());
            }
            ordered_features
        };

        let ordered_residual = if level == 0 {
            Vec::new()
        } else if level == 1 {
            if let Loss::LAD = loss {
                TrainingCache::cache_residual(data)
            } else {
                Vec::new()
            }
        } else {
            if let Loss::LAD = loss {
                cache.ordered_residual.to_vec()
            } else {
                Vec::new()
            }
        };

        SubCache {
            ordered_features,
            ordered_residual,
            lazy: false,
        }
    }

    fn get_empty() -> Self {
        SubCache {
            ordered_features: Vec::new(),
            ordered_residual: Vec::new(),
            lazy: false,
        }
    }

    pub fn split_cache(
        mut self,
        left_set: &[usize],
        right_set: &[usize],
        cache: &TrainingCache,
    ) -> (Self, Self) {
        if cache.cache_level == 0 {
            return (SubCache::get_empty(), SubCache::get_empty());
        }
        let mut left_ordered_features: Vec<Vec<(usize, ValueType)>> =
            Vec::with_capacity(cache.feature_size);
        let mut right_ordered_features: Vec<Vec<(usize, ValueType)>> =
            Vec::with_capacity(cache.feature_size);
        let mut left_ordered_residual = Vec::with_capacity(left_set.len());
        let mut right_ordered_residual = Vec::with_capacity(right_set.len());
        for index in 0..cache.feature_size {
            left_ordered_features.push(Vec::with_capacity(left_set.len()));
            right_ordered_features.push(Vec::with_capacity(right_set.len()));
        }
        let mut left_bool = vec![false; cache.sample_size];
        let mut right_bool = vec![false; cache.sample_size];
        for index in left_set.iter() {
            left_bool[*index] = true;
        }
        for index in right_set.iter() {
            right_bool[*index] = true;
        }

        if self.lazy {
            for (feature_index, feature_vec) in cache.ordered_features.iter().enumerate() {
                for pair in feature_vec.iter() {
                    let (index, value) = *pair;
                    if left_bool[index] {
                        left_ordered_features[feature_index].push((index, value));
                        continue;
                    }
                    if right_bool[index] {
                        right_ordered_features[feature_index].push((index, value));
                    }
                }
            }
        } else {
            for feature_index in 0..self.ordered_features.len() {
                let feature_vec = &mut self.ordered_features[feature_index];
                for pair in feature_vec.iter() {
                    let (index, value) = *pair;
                    if left_bool[index] {
                        left_ordered_features[feature_index].push((index, value));
                        continue;
                    }
                    if right_bool[index] {
                        right_ordered_features[feature_index].push((index, value));
                    }
                }
                feature_vec.clear();
                feature_vec.shrink_to_fit();
            }
            self.ordered_features.clear();
            self.ordered_features.shrink_to_fit();
        }

        if self.lazy {
            for pair in cache.ordered_residual.iter() {
                let (index, value) = *pair;
                if left_bool[index] {
                    left_ordered_residual.push((index, value));
                    continue;
                }
                if right_bool[index] {
                    right_ordered_residual.push((index, value));
                }
            }
        } else {
            for pair in self.ordered_residual.into_iter() {
                let (index, value) = pair;
                if left_bool[index] {
                    left_ordered_residual.push((index, value));
                    continue;
                }
                if right_bool[index] {
                    right_ordered_residual.push((index, value));
                }
            }
        }
        (
            SubCache {
                ordered_features: left_ordered_features,
                ordered_residual: left_ordered_residual,
                lazy: false,
            },
            SubCache {
                ordered_features: right_ordered_features,
                ordered_residual: right_ordered_residual,
                lazy: false,
            },
        )
    }

    /*
    fn sort_with_bool_vec(
        &self,
        feature_index: usize,
        is_residual: bool,
        to_sort: &[bool],
        to_sort_size: usize,
        sub_cache: &SubCache,
    ) -> Vec<(usize, ValueType)> {
        let whole_data_sorted_index = if is_residual {
            &self.ordered_residual
        } else {
            &self.ordered_features[feature_index]
        };
        let mut ret = Vec::with_capacity(to_sort_size);
        for item in whole_data_sorted_index.iter() {
            let (index, value) = *item;
            if to_sort[index] {
                ret.push((index, value));
            }
        }
        ret
    } */
}

/// A training sample or a test sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Data {
    /// the vector of features
    pub feature: Vec<ValueType>,
    /// the target value of the sample. Used in training.
    pub target: ValueType,
    /// sample weight. Used in training.
    pub weight: ValueType,
    /// sample's label. Used in training
    pub label: ValueType,
    /// used by LAD loss.
    pub residual: ValueType,
    /// used by gradient boost.
    pub initial_guess: ValueType,
}

/// The vector of the samples
pub type DataVec = Vec<Data>;
/// The vector of the predicted values.
pub type PredVec = Vec<ValueType>;

/*
pub enum Loss {
    SquaredError,
    LogLikelyhood,
    LAD,
}
*/

/// Calculate the prediction for each leaf node.
fn calculate_pred(
    data: &[usize],
    loss: &Loss,
    cache: &TrainingCache,
    sub_cache: &SubCache,
) -> ValueType {
    match loss {
        Loss::SquaredError => average(data, cache),
        Loss::LogLikelyhood => logit_optimal_value(data, cache),
        Loss::LAD => lad_optimal_value(data, cache, sub_cache),
        _ => average(data, cache),
    }
}

/// The leaf prediction value for SquaredError loss.
fn average(data: &[usize], cache: &TrainingCache) -> ValueType {
    let mut sum: f64 = 0.0;
    let mut weight: f64 = 0.0;

    for index in data.iter() {
        let cv: &CacheValue = &cache.cache_value[*index];
        sum += cv.s as f64;
        weight += cv.c as f64;
    }
    (sum / weight) as ValueType
}

/// The leaf prediction value for LogLikelyhood loss.
fn logit_optimal_value(data: &[usize], cache: &TrainingCache) -> ValueType {
    let mut s: f64 = 0.0;
    let mut c: f64 = 0.0;

    for index in data.iter() {
        s += cache.cache_value[*index].s as f64;
        c += cache.logit_c[*index] as f64;
    }

    if c.abs() < 1e-10 {
        0.0
    } else {
        (s / c) as ValueType
    }
}

/// The leaf prediction value for LAD loss.
fn lad_optimal_value(data: &[usize], cache: &TrainingCache, sub_cache: &SubCache) -> ValueType {
    let sorted_data = cache.sort_with_cache(0, true, data, sub_cache);

    let all_weight = sorted_data
        .iter()
        .fold(0.0f64, |acc, x| acc + (cache.cache_value[x.0].c) as f64);

    let mut weighted_median: f64 = 0.0;
    let mut weight: f64 = 0.0;
    for (i, pair) in sorted_data.iter().enumerate() {
        weight += cache.cache_value[pair.0].c as f64;
        if (weight * 2.0) > all_weight {
            if i >= 1 {
                weighted_median = ((pair.1 + sorted_data[i - 1].1) / 2.0) as f64;
            } else {
                weighted_median = pair.1 as f64;
            }

            break;
        }
    }
    weighted_median as ValueType
}

/// Return whether the data vector have same target values.
fn same(dv: &[Data], iv: &[usize]) -> bool {
    if iv.is_empty() {
        return false;
    }

    let t: ValueType = dv[iv[0]].target;
    for i in iv.iter().skip(1) {
        if !(almost_equal(t, dv[*i].target)) {
            return false;
        }
    }
    true
}

/// The internal node of the decision tree. It's stored in the `value` of the gbdt::binary_tree::BinaryTreeNode
///
#[derive(Debug, Serialize, Deserialize)]
struct DTNode {
    /// the feature used to split the node
    feature_index: usize,
    /// the feature value used to split the node
    feature_value: ValueType,
    /// the prediction of the leaf node
    pred: ValueType,
    /// how to handle missing value: -1 (left child), 0 (node prediction), 1 (right child)
    missing: i8,
    /// whether the node is a leaf node
    is_leaf: bool,
}

impl DTNode {
    /// Return an empty DTNode
    pub fn new() -> Self {
        DTNode {
            feature_index: 0,
            feature_value: 0.0,
            pred: 0.0,
            missing: 0,
            is_leaf: false,
        }
    }
}

/// The decision tree.
#[derive(Debug, Serialize, Deserialize)]
pub struct DecisionTree {
    /// the tree
    tree: BinaryTree<DTNode>,
    /// the size of feautures. Training data and test data should have same feature size.
    feature_size: usize,
    /// the max depth of the decision tree. The root node is considered to be in the layer 0.
    max_depth: u32,
    /// the minimum number of samples required to be at a leaf node during training.
    min_leaf_size: usize,
    /// the loss function type.
    loss: Loss,
    /// portion of features to be splited. When spliting a node, a subset of the features
    /// (feature_size * feature_sample_ratio) will be randomly selected to calculate impurity.
    feature_sample_ratio: f64,
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTree {
    /// Return a new decision tree with default values (feature_size = 1, max_depth = 2,
    /// min_leaf_size = 1, loss = Loss::SquaredError, feature_sample_ratio = 1.0)
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
    /// let mut tree = DecisionTree::new();
    /// ```
    pub fn new() -> Self {
        DecisionTree {
            tree: BinaryTree::new(),
            feature_size: 1,
            max_depth: 2,
            min_leaf_size: 1,
            loss: Loss::SquaredError,
            feature_sample_ratio: 1.0,
        }
    }

    /// Set the size of feautures. Training data and test data should have same feature size.
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
    /// let mut tree = DecisionTree::new();
    /// tree.set_feature_size(3);
    /// ```
    pub fn set_feature_size(&mut self, size: usize) {
        self.feature_size = size;
    }

    /// Set the max depth of the decision tree. The root node is considered to be in the layer 0.
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
    /// let mut tree = DecisionTree::new();
    /// tree.set_max_depth(2);
    /// ```
    pub fn set_max_depth(&mut self, max_depth: u32) {
        self.max_depth = max_depth;
    }

    /// Set the minimum number of samples required to be at a leaf node during training.
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
    /// let mut tree = DecisionTree::new();
    /// tree.set_min_leaf_size(1);
    /// ```
    pub fn set_min_leaf_size(&mut self, min_leaf_size: usize) {
        self.min_leaf_size = min_leaf_size;
    }

    /// Set the loss function type.
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
    /// let mut tree = DecisionTree::new();
    /// tree.set_loss(Loss::SquaredError);
    /// ```
    pub fn set_loss(&mut self, loss: Loss) {
        self.loss = loss;
    }

    /// Set the portion of features to be splited. When spliting a node, a subset of the features
    /// (feature_size * feature_sample_ratio) will be randomly selected to calculate impurity.
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
    /// let mut tree = DecisionTree::new();
    /// tree.set_feature_sample_ratio(0.9);
    /// ```
    pub fn set_feature_sample_ratio(&mut self, feature_sample_ratio: f64) {
        self.feature_sample_ratio = feature_sample_ratio;
    }

    /// Use the first `n` samples in the `train_data` to train a decision tree
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
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
    ///
    /// let mut dv = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    ///
    ///
    /// // train a decision tree
    /// let mut tree = DecisionTree::new();
    /// tree.set_feature_size(3);
    /// tree.set_max_depth(2);
    /// tree.set_min_leaf_size(1);
    /// tree.set_loss(Loss::SquaredError);
    /// let subset = [0, 1];
    /// let mut cache = TrainingCache::get_cache(3, &dv, 3);
    /// tree.fit_n(&dv, &subset, &mut cache);
    ///
    /// ```
    pub fn fit_n(&mut self, train_data: &DataVec, subset: &[usize], cache: &mut TrainingCache) {
        assert!(
            self.feature_size == cache.feature_size,
            "Decision_tree and TrainingCache should have same feature size"
        );

        cache.init_one_iteration(train_data, &self.loss);

        let root_index = self.tree.add_root(BinaryTreeNode::new(DTNode::new()));

        let sub_cache = SubCache::get_cache_from_training_cache(cache, train_data, &self.loss);

        self.fit_node(root_index, 0, subset, cache, sub_cache);
    }

    /// Use the samples in `train_data` to train the decision tree.
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
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
    ///
    /// let mut dv = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    ///
    ///
    /// // train a decision tree
    /// let mut tree = DecisionTree::new();
    /// tree.set_feature_size(3);
    /// tree.set_max_depth(2);
    /// tree.set_min_leaf_size(1);
    /// tree.set_loss(Loss::SquaredError);
    /// let mut cache = TrainingCache::get_cache(3, &dv, 3);
    /// tree.fit(&dv, &mut cache);
    ///
    /// ```
    pub fn fit(&mut self, train_data: &DataVec, cache: &mut TrainingCache) {
        //let mut gain: Vec<ValueType> = vec![0.0; self.feature_size];

        assert!(
            self.feature_size == cache.feature_size,
            "Decision_tree and TrainingCache should have same feature size"
        );
        let data_collection: Vec<usize> = (0..train_data.len()).collect();
        cache.init_one_iteration(train_data, &self.loss);

        let root_index = self.tree.add_root(BinaryTreeNode::new(DTNode::new()));
        let sub_cache = SubCache::get_cache_from_training_cache(cache, train_data, &self.loss);
        self.fit_node(root_index, 0, &data_collection, cache, sub_cache);
    }

    /// Recursively build the tree nodes. It choose a feature and a value to split the node and the data.
    /// And then use the splited data to build the child nodes.
    fn fit_node(
        &mut self,
        node: TreeIndex,
        depth: u32,
        //whole_data: &[Data],
        train_data: &[usize],
        cache: &mut TrainingCache,
        sub_cache: SubCache,
    ) {
        // If the node doesn't need to be splited.
        {
            let node_ref = self
                .tree
                .get_node_mut(node)
                .expect("node should not be empty!");
            // calculate to support unknown features
            node_ref.value.pred = calculate_pred(train_data, &self.loss, cache, &sub_cache);
            if (depth >= self.max_depth)
            //    || same(train_data)
                || (train_data.len() <= self.min_leaf_size)
            {
                node_ref.value.is_leaf = true;
                //node_ref.value.pred = calculate_pred(train_data, &self.loss);
                for index in train_data.iter() {
                    cache.preds[*index] = node_ref.value.pred;
                }
                return;
            }
        }

        // Try to find a feature and a value to split the node.
        let (splited_data, feature_index, feature_value) = DecisionTree::split(
            train_data,
            self.feature_size,
            self.feature_sample_ratio,
            cache,
            &sub_cache,
        );

        {
            let node_ref = self
                .tree
                .get_node_mut(node)
                .expect("node should not be empty");
            if splited_data.is_none() {
                node_ref.value.is_leaf = true;
                //node_ref.value.pred = calculate_pred(train_data, &self.loss);
                node_ref.value.pred = calculate_pred(train_data, &self.loss, cache, &sub_cache);
                for index in train_data.iter() {
                    cache.preds[*index] = node_ref.value.pred;
                }
                return;
            } else {
                node_ref.value.feature_index = feature_index;
                node_ref.value.feature_value = feature_value;
            }
        }

        // Use the splited data to build child nodes.
        if let Some((left_data, right_data, _unknown_data)) = splited_data {
            let (left_sub_cache, right_sub_cache) =
                sub_cache.split_cache(&left_data, &right_data, cache);
            let left_index = self
                .tree
                .add_left_node(node, BinaryTreeNode::new(DTNode::new()));
            self.fit_node(left_index, depth + 1, &left_data, cache, left_sub_cache);
            let right_index = self
                .tree
                .add_right_node(node, BinaryTreeNode::new(DTNode::new()));
            self.fit_node(right_index, depth + 1, &right_data, cache, right_sub_cache);
        }
    }

    /// Inference the values of the first `n` samples in the `test_data`. Return a vector of
    /// predicted values.
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
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
    /// weight: 1.0,
    /// label: 0.0,
    /// residual: 0.0,
    /// initial_guess: 1.0,
    /// };
    ///
    /// let mut dv = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    ///
    /// // train a decision tree
    /// let mut tree = DecisionTree::new();
    /// tree.set_feature_size(3);
    /// tree.set_max_depth(2);
    /// tree.set_min_leaf_size(1);
    /// tree.set_loss(Loss::SquaredError);
    /// let mut cache = TrainingCache::get_cache(3, &dv, 3);
    /// tree.fit(&dv, &mut cache);
    ///
    ///
    /// // set up the test data
    /// let mut dv = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    ///
    /// // inference the test data with the decision tree
    /// let subset = [0,1,2];
    /// println!("{:?}", tree.predict_n(&dv, &subset));
    ///
    ///
    /// // output:
    /// // [2.0, 0.75, 0.75]
    /// ```
    ///
    /// # Panic
    /// If the function is called before the decision tree is trained, it will panic.
    ///
    /// If the test data have a smaller feature size than the tree's feature size, it will panic.
    pub fn predict_n(&self, test_data: &DataVec, subset: &[usize]) -> PredVec {
        let root = self
            .tree
            .get_node(self.tree.get_root_index())
            .expect("Decision tree should have root node");

        let mut ret = vec![0.0; test_data.len()];
        // Inference the samples one by one.
        for index in subset {
            ret[*index] = self.predict_one(root, &test_data[*index]);
        }
        ret
    }

    /// Inference the values of samples in the `test_data`. Return a vector of the predicted
    /// values.
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
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
    /// weight: 1.0,
    /// label: 0.0,
    /// residual: 0.0,
    /// initial_guess: 1.0,
    /// };
    ///
    /// let mut dv = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    ///
    /// // train a decision tree
    /// let mut tree = DecisionTree::new();
    /// tree.set_feature_size(3);
    /// tree.set_max_depth(2);
    /// tree.set_min_leaf_size(1);
    /// tree.set_loss(Loss::SquaredError);
    /// let mut cache = TrainingCache::get_cache(3, &dv, 3);
    /// tree.fit(&dv, &mut cache);
    ///
    ///
    /// // set up the test data
    /// let mut dv = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    /// dv.push(data4.clone());
    ///
    ///
    /// // inference the test data with the decision tree
    /// println!("{:?}", tree.predict(&dv));
    ///
    ///
    /// // output:
    /// // [2.0, 0.75, 0.75, 3.0]
    /// ```
    /// # Panic
    /// If the function is called before the decision tree is trained, it will panic.
    ///
    /// If the test data have a smaller feature size than the tree's feature size, it will panic.
    pub fn predict(&self, test_data: &DataVec) -> PredVec {
        let root = self
            .tree
            .get_node(self.tree.get_root_index())
            .expect("Decision tree should have root node");

        // Inference the data one by one
        test_data
            .iter()
            .map(|x| self.predict_one(root, x))
            .collect()
    }

    /// Inference a `sample` from current `node`
    /// If the current node is a leaf node, return the node's prediction. Otherwise, choose a child
    /// node according to the feature and feature value of the node. Then call this function recursively.
    fn predict_one(&self, node: &BinaryTreeNode<DTNode>, sample: &Data) -> ValueType {
        let mut is_node_value = false;
        let mut is_left_child = false;
        let mut _is_right_child = false;
        if node.value.is_leaf {
            is_node_value = true;
        } else {
            assert!(
                sample.feature.len() > node.value.feature_index,
                "sample doesn't have the feature"
            );

            if sample.feature[node.value.feature_index] == VALUE_TYPE_UNKNOWN {
                if node.value.missing == -1 {
                    is_left_child = true;
                } else if node.value.missing == 0 {
                    is_node_value = true;
                } else {
                    _is_right_child = true;
                }
            } else if sample.feature[node.value.feature_index] < node.value.feature_value {
                is_left_child = true;
            } else {
                _is_right_child = true;
            }
        }

        // return the node's prediction
        if is_node_value {
            node.value.pred
        } else if is_left_child {
            let left = self
                .tree
                .get_left_child(node)
                .expect("Left child should not be None");
            self.predict_one(left, sample)
        } else {
            let right = self
                .tree
                .get_right_child(node)
                .expect("Right child should not be None");
            self.predict_one(right, sample)
        }
    }

    /// Split the data by calculating the impurity.
    /// Step 1: Choose candidate features. If `feature_sample_ratio` < 1.0, randomly selected
    /// (feature_sample_ratio * feature_size) features. Otherwise, choose all features.
    ///
    /// Step 2: Calculate each feature's impurity and the corresponding value to split the data.
    ///
    /// Step 3: Find the feature that has the smallest impurity.
    ///
    /// Step 4: Use the feature and the feature value to split the data.
    fn split(
        //whole_data: &[Data],
        train_data: &[usize],
        feature_size: usize,
        feature_sample_ratio: f64,
        cache: &TrainingCache,
        sub_cache: &SubCache,
    ) -> (
        Option<(Vec<usize>, Vec<usize>, Vec<usize>)>,
        usize,
        ValueType,
    ) {
        let mut fs = feature_size;
        let mut fv: Vec<usize> = (0..).take(fs).collect();

        let mut rng = thread_rng();
        if feature_sample_ratio < 1.0 {
            fs = (feature_sample_ratio * (feature_size as f64)) as usize;
            fv.shuffle(&mut rng);
        }

        let mut v: ValueType = 0.0;
        let mut impurity: f64 = 0.0;
        //let mut g: f64 = 0.0;
        let mut best_fitness: f64 = std::f64::MAX;

        let mut index: usize = 0;
        let mut value: ValueType = 0.0;
        // let mut gain: f64 = 0.0;

        let mut impurity_cache = ImpurityCache::new(cache.sample_size, train_data);

        let mut find: bool = false;
        let mut data_to_split: Vec<(usize, ValueType)> = Vec::new();
        for i in fv.iter().take(fs) {
            let sorted_data = DecisionTree::get_impurity(
                train_data,
                *i,
                &mut v,
                &mut impurity,
                cache,
                &mut impurity_cache,
                &sub_cache,
            );
            if best_fitness > impurity {
                find = true;
                best_fitness = impurity;
                index = *i;
                value = v;
                data_to_split = sorted_data;
                //gain = g;
            }
        }
        if find {
            let mut left: Vec<usize> = Vec::new();
            let mut right: Vec<usize> = Vec::new();
            let mut unknown: Vec<usize> = Vec::new();
            for pair in data_to_split.iter() {
                let (index, feature_value) = *pair;
                if feature_value == VALUE_TYPE_UNKNOWN {
                    unknown.push(index);
                } else if feature_value < value {
                    left.push(index);
                } else {
                    right.push(index);
                }
            }
            (Some((left, right, unknown)), index, value)
        } else {
            (None, 0, 0.0)
        }
    }

    /// Calculate the impurity.
    fn get_impurity(
        //_whole_data: &[Data],
        train_data: &[usize],
        feature_index: usize,
        value: &mut ValueType,
        impurity: &mut f64,
        cache: &TrainingCache,
        impurity_cache: &mut ImpurityCache,
        sub_cache: &SubCache,
        //gain: &mut f64,
    ) -> Vec<(usize, ValueType)> {
        *impurity = std::f64::MAX;
        *value = VALUE_TYPE_UNKNOWN;
        let sorted_data = cache.sort_with_bool_vec(
            feature_index,
            false,
            &impurity_cache.bool_vec,
            impurity_cache.sample_size,
            sub_cache,
        );

        let mut unknown: usize = 0;
        let mut s: f64 = 0.0;
        let mut ss: f64 = 0.0;
        let mut c: f64 = 0.0;

        for pair in sorted_data.iter() {
            let (index, feature_value) = *pair;
            if feature_value == VALUE_TYPE_UNKNOWN {
                let cv: &CacheValue = &cache.cache_value[index];
                s += cv.s as f64;
                ss += cv.ss as f64;
                c += cv.c as f64;
                unknown += 1;
            } else {
                break;
            }
        }

        if unknown == sorted_data.len() {
            return sorted_data;
        }

        let mut fitness0 = if c > 1.0 { ss - s * s / c } else { 0.0 };

        if fitness0 < 0.0 {
            fitness0 = 0.0;
        }

        if !impurity_cache.cached {
            impurity_cache.sum_s = 0.0;
            impurity_cache.sum_ss = 0.0;
            impurity_cache.sum_c = 0.0;
            for index in train_data.iter() {
                let cv: &CacheValue = &cache.cache_value[*index];
                impurity_cache.sum_s += cv.s as f64;
                impurity_cache.sum_ss += cv.ss as f64;
                impurity_cache.sum_c += cv.c as f64;
            }
        }
        s = impurity_cache.sum_s - s;
        ss = impurity_cache.sum_ss - ss;
        c = impurity_cache.sum_c - c;

        let _fitness00: f64 = if c > 1.0 { ss - s * s / c } else { 0.0 };

        let mut ls: f64 = 0.0;
        let mut lss: f64 = 0.0;
        let mut lc: f64 = 0.0;
        let mut rs: f64 = s;
        let mut rss: f64 = ss;
        let mut rc: f64 = c;

        for i in unknown..(sorted_data.len() - 1) {
            let (index, feature_value) = sorted_data[i];
            let (_next_index, next_value) = sorted_data[i + 1];
            let cv: &CacheValue = &cache.cache_value[index];
            s = cv.s as f64;
            ss = cv.ss as f64;
            c = cv.c as f64;

            ls += s;
            lss += ss;
            lc += c;

            rs -= s;
            rss -= ss;
            rc -= c;

            let f1: ValueType = feature_value;
            let f2: ValueType = next_value;

            if almost_equal(f1, f2) {
                continue;
            }

            let mut fitness1: f64 = if lc > 1.0 { lss - ls * ls / lc } else { 0.0 };
            if fitness1 < 0.0 {
                fitness1 = 0.0;
            }

            let mut fitness2: f64 = if rc > 1.0 { rss - rs * rs / rc } else { 0.0 };
            if fitness2 < 0.0 {
                fitness2 = 0.0;
            }

            let fitness: f64 = fitness0 + fitness1 + fitness2;

            if *impurity > fitness {
                *impurity = fitness;
                *value = (f1 + f2) / 2.0;
                //*gain = fitness00 - fitness1 - fitness2;
            }
        }

        sorted_data
    }

    /// Print the decision tree. For debug use.
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
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
    ///
    /// let mut dv = Vec::new();
    /// dv.push(data1.clone());
    /// dv.push(data2.clone());
    /// dv.push(data3.clone());
    ///
    ///
    /// // train a decision tree
    /// let mut tree = DecisionTree::new();
    /// tree.set_feature_size(3);
    /// tree.set_max_depth(2);
    /// tree.set_min_leaf_size(1);
    /// tree.set_loss(Loss::SquaredError);
    /// let mut cache = TrainingCache::get_cache(3, &dv, 3);
    /// let subset = [0, 1];
    /// tree.fit_n(&dv, &subset, &mut cache);
    ///
    ///
    /// tree.print();
    ///
    /// // output:
    ///
    /// //  ----DTNode { feature_index: 0, feature_value: 1.05, pred: 0.0, is_leaf: false }
    /// //      ----DTNode { feature_index: 0, feature_value: 0.0, pred: 2.0, is_leaf: true }
    /// //      ----DTNode { feature_index: 0, feature_value: 0.0, pred: 1.0, is_leaf: true }
    /// ```
    pub fn print(&self) {
        self.tree.print();
    }

    pub fn get_from_xgboost(node: &serde_json::Value) -> Result<Self, Box<Error>> {
        // Parameters are not used in prediction process, so we use default parameters.
        let mut tree = DecisionTree::new();
        let index = tree.tree.add_root(BinaryTreeNode::new(DTNode::new()));
        tree.add_node_from_json(index, node)?;
        Ok(tree)
    }

    fn add_node_from_json(
        &mut self,
        index: TreeIndex,
        node: &serde_json::Value,
    ) -> Result<(), Box<Error>> {
        {
            let node_ref = self
                .tree
                .get_node_mut(index)
                .expect("node should not be empty!");
            if let serde_json::Value::Number(pred) = &node["leaf"] {
                let leaf_value = pred.as_f64().ok_or("parse 'leaf' error")?;
                node_ref.value.pred = leaf_value as ValueType;
                node_ref.value.is_leaf = true;
                return Ok(());
            } else {
                let feature_value = node["split_condition"]
                    .as_f64()
                    .ok_or("parse 'split condition' error")?;
                node_ref.value.feature_value = feature_value as ValueType;

                let feature_index = match node["split"].as_i64() {
                    Some(v) => v,
                    None => {
                        let feature_name = node["split"].as_str().ok_or("parse 'split' error")?;
                        let feature_str: String = feature_name.chars().skip(3).collect();
                        feature_str.parse::<i64>()?
                    }
                };
                node_ref.value.feature_index = feature_index as usize;

                let missing = node["missing"].as_i64().ok_or("parse 'missing' error")?;
                let left_child = node["yes"].as_i64().ok_or("parse 'yes' error")?;
                let right_child = node["no"].as_i64().ok_or("parse 'no' error")?;
                if missing == left_child {
                    node_ref.value.missing = -1;
                } else if missing == right_child {
                    node_ref.value.missing = 1;
                } else {
                    let err: Box<Error> = From::from("not support extra missing node".to_string());
                    return Err(err);
                }
            }
        }

        let left_child = node["yes"].as_i64().ok_or("parse 'yes' error")?;
        let right_child = node["no"].as_i64().ok_or("parse 'no' error")?;
        let children = node["children"]
            .as_array()
            .ok_or("parse 'children' error")?;
        let mut find_left = false;
        let mut find_right = false;
        for child in children.iter() {
            let node_id = child["nodeid"].as_i64().ok_or("parse 'nodeid' error")?;
            if node_id == left_child {
                find_left = true;
                let left_index = self
                    .tree
                    .add_left_node(index, BinaryTreeNode::new(DTNode::new()));
                self.add_node_from_json(left_index, child)?;
            }
            if node_id == right_child {
                find_right = true;
                let right_index = self
                    .tree
                    .add_right_node(index, BinaryTreeNode::new(DTNode::new()));
                self.add_node_from_json(right_index, child)?;
            }
        }

        if (!find_left) || (!find_right) {
            let err: Box<Error> = From::from("children not found".to_string());
            return Err(err);
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }
}
