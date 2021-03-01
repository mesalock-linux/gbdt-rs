//! This module implements a decision tree from the simple binary tree [gbdt::binary_tree].
//!
//! In the training process, the nodes are splited according `impurity`.
//!
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
//! let data1 = Data::new_training_data(
//!     vec![1.0, 2.0, 3.0],
//!     1.0,
//!     2.0,
//!     None
//! );
//! let data2 = Data::new_training_data(
//!     vec![1.1, 2.1, 3.1],
//!     1.0,
//!     1.0,
//!     None
//! );
//! let data3 = Data::new_training_data(
//!     vec![2.0, 2.0, 1.0],
//!     1.0,
//!     0.5,
//!     None
//! );
//! let data4 = Data::new_training_data(
//!     vec![2.0, 2.3, 1.2],
//!     1.0,
//!     3.0,
//!     None,
//! );
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
//! let mut cache = TrainingCache::get_cache(3, &dv, 2);
//! tree.fit(&dv, &mut cache);
//!
//!
//! // set up the test data
//! let mut dv = Vec::new();
//! dv.push(data1.clone());
//! dv.push(data2.clone());
//! dv.push(Data::new_test_data(
//!     vec![2.0, 2.0, 1.0],
//!     None));
//! dv.push(Data::new_test_data(
//!     vec![2.0, 2.3, 1.2],
//!     Some(3.0)));
//!
//!
//! // inference the test data with the decision tree
//! println!("{:?}", tree.predict(&dv));
//!
//!
//! // output:
//! // [2.0, 0.75, 0.75, 3.0]
//! ```

#[cfg(all(feature = "mesalock_sgx", not(target_env = "sgx")))]
use std::prelude::v1::*;

use crate::binary_tree::BinaryTree;
use crate::binary_tree::BinaryTreeNode;
use crate::binary_tree::TreeIndex;
use crate::config::Loss;
#[cfg(feature = "enable_training")]
use crate::fitness::almost_equal;
use std::error::Error;

#[cfg(feature = "enable_training")]
use rand::prelude::SliceRandom;
#[cfg(feature = "enable_training")]
use rand::thread_rng;

use serde_derive::{Deserialize, Serialize};

///! For now we only support std::$t using this macro.
/// We will generalize ValueType in future.
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

/// A training sample or a test sample. You can call `new_training_data` to generate a training sample, and call `new_test_data` to generate a test sample.
///
/// A training sample can be used as a test sample.
///
/// You can also directly generate a data with following guides:
///
/// 1. When using the gbdt algorithm for training, you should set the values of feature, weight and label. If Config::initial_guess_enabled is true, you should set the value of initial_guess as well. Other fields can be arbitrary value.
///
/// 2. When using the gbdt algorithm for inference, you should set the value of feature. Other fields can be arbitrary value.
///
/// 3. When directly using the decision tree for training, only "SquaredError" is supported and you should set the values of feature, weight, label and target. `label` and `target` are equal. Other fields can be arbitrary value.
///
/// 4. When directly using the decision tree for inference, only "SquaredError" is supported and you should set the values of feature.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Data {
    /// the vector of features
    pub feature: Vec<ValueType>,
    /// the target value of the sample to be fit in one decistion tree. This value is calculated by gradient boost algorithm. If you want to use the decision tree with "SquaredError" directly, set this value with `label` value    
    pub target: ValueType,
    /// sample's weight. Used in training.
    pub weight: ValueType,
    /// sample's label. Used in training. This value is the actual value of the training sample.
    pub label: ValueType,
    /// used by LAD loss. Calculated by gradient boost algorithm.
    pub residual: ValueType,
    /// used by gradient boost. Set this value if Config::initial_guess_enabled is true.
    pub initial_guess: ValueType,
}

impl Data {
    /// Generate a training sample.
    ///
    /// feature: the vector of features
    ///
    /// weight: sample's weight
    ///
    /// label: sample's label
    ///
    /// initial_guess: initial prediction for the sample. This value is optional. Set this value if Config::initial_guess_enabled is true.
    ///
    /// # Example
    /// ``` rust
    /// use gbdt::decision_tree::Data;
    /// let data1 = Data::new_training_data(vec![1.0, 2.0, 3.0],
    ///                                 1.0,
    ///                                 2.0,
    ///                                 Some(0.5));
    /// let data2 = Data::new_training_data(vec![1.0, 2.0, 3.0],
    ///                                 1.0,
    ///                                 2.0,
    ///                                 None);
    /// ```
    pub fn new_training_data(
        feature: Vec<ValueType>,
        weight: ValueType,
        label: ValueType,
        initial_guess: Option<ValueType>,
    ) -> Self {
        Data {
            feature,
            target: label,
            weight,
            label,
            residual: label,
            initial_guess: initial_guess.unwrap_or(0.0),
        }
    }

    /// Generate a test sample.
    ///
    /// label: sample's label. It's optional.
    ///
    /// # Example
    /// ``` rust
    /// use gbdt::decision_tree::Data;
    /// let data1 = Data::new_test_data(vec![1.0, 2.0, 3.0],
    ///                                 Some(0.5));
    /// let data2 = Data::new_test_data(vec![1.0, 2.0, 3.0],
    ///                                 None);
    /// ```
    pub fn new_test_data(feature: Vec<ValueType>, label: Option<ValueType>) -> Self {
        Data {
            feature,
            target: 0.0,
            weight: 1.0,
            label: label.unwrap_or(0.0),
            residual: 0.0,
            initial_guess: 0.0,
        }
    }
}

/// The vector of the samples
pub type DataVec = Vec<Data>;
/// The vector of the predicted values.
pub type PredVec = Vec<ValueType>;

/// Cache some values for calculating the impurity.
#[cfg(feature = "enable_training")]
struct ImpurityCache {
    /// sum of target * weight
    sum_s: f64,
    /// sum of target * target * weight
    sum_ss: f64,
    /// sum of weight
    sum_c: f64,
    /// whether this cache is calcualted
    cached: bool,
    /// whether a data is in the current node
    bool_vec: Vec<bool>,
}

#[cfg(feature = "enable_training")]
impl ImpurityCache {
    fn new(sample_size: usize, train_data: &[usize]) -> Self {
        let mut bool_vec: Vec<bool> = vec![false; sample_size];
        // set bool_vec
        for index in train_data.iter() {
            bool_vec[*index] = true;
        }
        ImpurityCache {
            sum_s: 0.0,
            sum_ss: 0.0,
            sum_c: 0.0,
            cached: false, //`cached` is false
            bool_vec,
        }
    }
}

/// These results are repeatly used together: target*weight, target*target*weight, weight
#[cfg(feature = "enable_training")]
struct CacheValue {
    /// target * weight
    s: f64,
    /// target * target * weight
    ss: f64,
    /// weight
    c: f64,
}

/// Cache the sort results and some calculation results
#[cfg(feature = "enable_training")]
pub struct TrainingCache {
    /// Sort the training data with the feature value.
    /// ordered_features[i] is the data sorted by (i+1)th feature.
    /// (usize, ValueType) is the sample's index in the training set and its (i+1)th feature value.
    ordered_features: Vec<Vec<(usize, ValueType)>>,
    /// Sort the training data with the residual field.
    /// (uisze, ValueType) is the smaple's index in the training set and its residual value.
    ordered_residual: Vec<(usize, ValueType)>,
    /// cache_value[i] is the (i+1)th sample's `CacheValue`
    cache_value: Vec<CacheValue>, //s, ss, c
    /// cache_target[i] is the (i+1)th sample's `target` value (not the label). Organizing the `target` and `CacheValue` together will have better spatial locality.
    cache_target: Vec<ValueType>,
    /// loigt_c[i] is the (i+1)th sample's logit value. let y = target.abs(); let logit_value = y * (2.0 - y) * weight;
    logit_c: Vec<ValueType>,
    /// The sample size of the training set.
    sample_size: usize,
    /// The feature size of the training data
    feature_size: usize,
    /// The prediction of the training samples.
    preds: Vec<ValueType>,
    /// The cache level.
    /// 0: ordered_features is calculated only once. SubCache is not used.
    /// 1: ordered_features is calculated in each iterations. SubCache is used.
    /// 2: ordered_features is calculated only once. SubCache is used.
    cache_level: u8,
}

#[cfg(feature = "enable_training")]
impl TrainingCache {
    /// Allocate the training cache. Feature size, training set and cache level should be provided.
    /// ``` rust
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
    /// let data1 = Data::new_training_data(
    ///     vec![1.0, 2.0, 3.0],
    ///     1.0,
    ///     2.0,
    ///     None
    /// );
    /// let data2 = Data::new_training_data(
    ///     vec![1.1, 2.1, 3.1],
    ///     1.0,
    ///     1.0,
    ///     None
    /// );
    /// let data3 = Data::new_training_data(
    ///     vec![2.0, 2.0, 1.0],
    ///     1.0,
    ///     0.5,
    ///     None
    /// );
    /// let mut dv = Vec::new();
    /// dv.push(data1);
    /// dv.push(data2);
    /// dv.push(data3);
    ///
    /// let mut cache = TrainingCache::get_cache(3, &dv, 2);
    /// ```
    // Only ordered_features may be pre-computed. Other fields will be computed by calling init_one_iteration
    pub fn get_cache(feature_size: usize, data: &DataVec, cache_level: u8) -> Self {
        // cache_level is 0, 1 or 2.
        let level = if cache_level >= 3 { 2 } else { cache_level };

        let sample_size = data.len();
        let logit_c = vec![0.0; data.len()];
        let preds = vec![VALUE_TYPE_UNKNOWN; sample_size];

        let mut cache_value = Vec::with_capacity(data.len());
        for elem in data {
            let item = CacheValue {
                s: 0.0,
                ss: 0.0,
                c: f64::from(elem.weight),
            };
            cache_value.push(item);
        }

        // Calculate the ordred_features if cache_level is 0 or 2.
        let ordered_features: Vec<Vec<(usize, ValueType)>> = if (level == 0) || (level == 2) {
            TrainingCache::cache_features(data, feature_size)
        } else {
            Vec::new()
        };

        let ordered_residual: Vec<(usize, ValueType)> = Vec::new();

        let cache_target: Vec<ValueType> = vec![0.0; data.len()];

        TrainingCache {
            ordered_features,
            ordered_residual,
            cache_value,
            cache_target,
            logit_c,
            sample_size,
            feature_size,
            preds,
            cache_level: level,
        }
    }

    /// Return the training data's predictions using this decision tree. These results are computed during training and then cached.
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
    /// let data1 = Data::new_training_data(
    ///     vec![1.0, 2.0, 3.0],
    ///     1.0,
    ///     2.0,
    ///     None
    /// );
    /// let data2 = Data::new_training_data(
    ///     vec![1.1, 2.1, 3.1],
    ///     1.0,
    ///     1.0,
    ///     None
    /// );
    /// let data3 = Data::new_training_data(
    ///     vec![2.0, 2.0, 1.0],
    ///     1.0,
    ///     0.5,
    ///     None
    /// );
    /// let data4 = Data::new_training_data(
    ///     vec![2.0, 2.3, 1.2],
    ///     1.0,
    ///     3.0,
    ///     None
    /// );
    ///
    /// let mut dv = Vec::new();
    /// dv.push(data1);
    /// dv.push(data2);
    /// dv.push(data3);
    /// dv.push(data4);
    ///
    ///
    /// // train a decision tree
    /// let mut tree = DecisionTree::new();
    /// tree.set_feature_size(3);
    /// tree.set_max_depth(2);
    /// tree.set_min_leaf_size(1);
    /// tree.set_loss(Loss::SquaredError);
    /// let mut cache = TrainingCache::get_cache(3, &dv, 2);
    /// tree.fit(&dv, &mut cache);
    /// // get predictions for the training data
    /// println!("{:?}", cache.get_preds());
    ///
    ///
    /// // output:
    /// // [2.0, 0.75, 0.75, 3.0]
    /// ```
    pub fn get_preds(&self) -> Vec<ValueType> {
        self.preds.to_vec()
    }

    /// Compute the training cache in the begining of training the deceision tree.
    ///
    /// `whole_data`: the training set.
    ///
    /// `loss`: the loss type.  
    fn init_one_iteration(&mut self, whole_data: &[Data], loss: &Loss) {
        // Compute the cache_target, cache_value, logit_c.
        for (index, data) in whole_data.iter().enumerate() {
            let target = data.target;
            self.cache_target[index] = target;
            let weight = f64::from(data.weight);
            let target = f64::from(target);
            let s = target * weight;
            self.cache_value[index].s = s;
            self.cache_value[index].ss = target * s;
            if let Loss::LogLikelyhood = loss {
                let y = target.abs();
                let c = y * (2.0 - y) * weight;
                self.logit_c[index] = c as ValueType;
            }
        }
        // Compute the ordered_residual.
        if let Loss::LAD = loss {
            self.ordered_residual = TrainingCache::cache_residual(whole_data);
        }
    }

    /// Compute the ordered_features.
    ///
    /// Input the training set (`whole_data`) and feature size (`feature_size`)
    ///
    /// Output the ordered_features.
    fn cache_features(whole_data: &[Data], feature_size: usize) -> Vec<Vec<(usize, ValueType)>> {
        // Allocate memory
        let mut ordered_features = Vec::with_capacity(feature_size);
        for _index in 0..feature_size {
            let nv: Vec<(usize, ValueType)> = Vec::with_capacity(whole_data.len());
            ordered_features.push(nv);
        }

        // Put data
        for (i, item) in whole_data.iter().enumerate() {
            for (index, ordered_item) in ordered_features.iter_mut().enumerate().take(feature_size)
            {
                ordered_item.push((i, item.feature[index]));
            }
        }

        // Sort all the vectors
        for item in ordered_features.iter_mut().take(feature_size) {
            item.sort_unstable_by(|a, b| {
                let v1 = a.1;
                let v2 = b.1;
                v1.partial_cmp(&v2).unwrap()
            });
        }

        ordered_features
    }

    /// Compute the ordered_residual.
    ///
    /// Input the training set (`whole_data`). Output the ordered_residual.
    fn cache_residual(whole_data: &[Data]) -> Vec<(usize, ValueType)> {
        // Allocate memory
        let mut ordered_residual = Vec::with_capacity(whole_data.len());

        // Put data.
        for (index, elem) in whole_data.iter().enumerate() {
            ordered_residual.push((index, elem.residual));
        }

        // Sort data
        ordered_residual.sort_unstable_by(|a, b| {
            let v1: ValueType = a.1;
            let v2: ValueType = b.1;
            v1.partial_cmp(&v2).unwrap()
        });

        ordered_residual
    }

    /// Sort data with Training Cache. Bucket sort is used.
    ///
    /// feature_index: which feature is used to sort.
    ///
    /// is_residual: true, sort with residual value; false, sort with feature value
    ///
    /// to_sort: bool vector. The index is the sample's index in whole training set. The boolean value indicates whether the sample is needed to be sorted.
    ///
    /// to_sort_size: the amount of the data.
    ///
    /// sub_cache: SubCache.
    fn sort_with_bool_vec(
        &self,
        feature_index: usize,
        is_residual: bool,
        to_sort: &[bool],
        to_sort_size: usize,
        sub_cache: &SubCache,
    ) -> Vec<(usize, ValueType)> {
        // Get sorted data.
        let whole_data_sorted_index = if is_residual {
            if (self.cache_level == 0) || sub_cache.lazy {
                &self.ordered_residual
            } else {
                &sub_cache.ordered_residual
            }
        } else if (self.cache_level == 0) || sub_cache.lazy {
            &self.ordered_features[feature_index]
        } else {
            &sub_cache.ordered_features[feature_index]
        };

        // The whole_data_sorted_index.len() is greater than or equal to to_sort_size. If they are equal, then whole_data_sorted_index is what we want.
        if whole_data_sorted_index.len() == to_sort_size {
            return whole_data_sorted_index.to_vec();
        }

        // Filter the whole_data_sorted_index with the boolean vector.
        let mut ret = Vec::with_capacity(to_sort_size);
        for item in whole_data_sorted_index.iter() {
            let (index, value) = *item;
            if to_sort[index] {
                ret.push((index, value));
            }
        }
        ret
    }

    /// Sort data with Training Cache. Bucket sort is used.
    ///
    /// feature_index: which feature is used to sort.
    ///
    /// is_residual: true, sort with residual value; false, sort with feature value
    ///
    /// to_sort: a vector containing samples' indexes.
    ///
    /// sub_cache: SubCache.
    fn sort_with_cache(
        &self,
        feature_index: usize,
        is_residual: bool,
        to_sort: &[usize],
        sub_cache: &SubCache,
    ) -> Vec<(usize, ValueType)> {
        // Allocate the boolean vector
        let whole_data_sorted_index = if is_residual {
            &self.ordered_residual
        } else {
            &self.ordered_features[feature_index]
        };
        let mut index_exists: Vec<bool> = vec![false; whole_data_sorted_index.len()];

        // Generate the boolean vector
        for index in to_sort.iter() {
            index_exists[*index] = true;
        }

        // Call sort_with_bool_vec to get sorted data
        self.sort_with_bool_vec(
            feature_index,
            is_residual,
            &index_exists,
            to_sort.len(),
            sub_cache,
        )
    }
}

/// SubCache is used to accelerate the data sorting.
/// ordered_features and ordered_residual are used in bucket sort.
/// In TrainingCache, the two vectors contains information from the whole training set. But only the information from samples in current node are needed.
/// So SubCache only restore the information from samples in current node.
#[cfg(feature = "enable_training")]
struct SubCache {
    /// Sort the samples with the feature value.
    /// ordered_features[i] is the data sorted by (i+1)th feature.
    /// (usize, ValueType) is the sample's index in the whole training set and its (i+1)th feature value.
    ordered_features: Vec<Vec<(usize, ValueType)>>,
    /// Sort the samples with the residual field.
    /// (uisze, ValueType) is the smaple's index in the whole training set and its residual value.
    ordered_residual: Vec<(usize, ValueType)>,
    /// True means the SubCache is not computed. For the root node, the samples in current node are the whole training set. So SubCache is not needed.
    lazy: bool,
}
#[cfg(feature = "enable_training")]
impl SubCache {
    /// Generate the SubCache frome the TrainingCache. `data` is the whole training set. `loss` is the loss type.
    fn get_cache_from_training_cache(cache: &TrainingCache, data: &[Data], loss: &Loss) -> Self {
        let level = cache.cache_level;

        // level 2: lazy is True, ordered_features and ordered_residual is empty.
        if level == 2 {
            return SubCache {
                ordered_features: Vec::new(),
                ordered_residual: Vec::new(),
                lazy: true,
            };
        }

        // level 0: ordered_features is empty, lazy is false.
        let ordered_features = if level == 0 {
            Vec::new()
        } else if level == 1 {
            // level 1: ordered_features is computed from the whole training set. lazy is false
            TrainingCache::cache_features(data, cache.feature_size)
        } else {
            // Other: clone the ordered_features in the TrainingCache.
            let mut ordered_features: Vec<Vec<(usize, ValueType)>> =
                Vec::with_capacity(cache.feature_size);

            for index in 0..cache.feature_size {
                ordered_features.push(cache.ordered_features[index].to_vec());
            }
            ordered_features
        };

        // level 0: ordered_residual is empty. lazy is false.
        let ordered_residual = if level == 0 {
            Vec::new()
        } else if level == 1 {
            // level 1: ordered_residual is computed from the whole train set. lazy if alse
            if let Loss::LAD = loss {
                TrainingCache::cache_residual(data)
            } else {
                Vec::new()
            }
        } else {
            // other: clone the ordered_features in the TrainingCache.
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

    /// Generate an empty SubCache.
    fn get_empty() -> Self {
        SubCache {
            ordered_features: Vec::new(),
            ordered_residual: Vec::new(),
            lazy: false,
        }
    }

    /// Split the SubCache for child nodes
    ///  
    /// left_set: the samples in left child.
    ///
    /// right_set: the samples in right child.
    ///
    /// cache: the TrainingCache
    ///
    /// output: two SubCache
    fn split_cache(
        mut self,
        left_set: &[usize],
        right_set: &[usize],
        cache: &TrainingCache,
    ) -> (Self, Self) {
        // level 0: return empty SubCache
        if cache.cache_level == 0 {
            return (SubCache::get_empty(), SubCache::get_empty());
        }

        // allocate the vectors
        let mut left_ordered_features: Vec<Vec<(usize, ValueType)>> =
            Vec::with_capacity(cache.feature_size);
        let mut right_ordered_features: Vec<Vec<(usize, ValueType)>> =
            Vec::with_capacity(cache.feature_size);
        let mut left_ordered_residual = Vec::with_capacity(left_set.len());
        let mut right_ordered_residual = Vec::with_capacity(right_set.len());
        for _ in 0..cache.feature_size {
            left_ordered_features.push(Vec::with_capacity(left_set.len()));
            right_ordered_features.push(Vec::with_capacity(right_set.len()));
        }

        // compute two boolean vectors
        let mut left_bool = vec![false; cache.sample_size];
        let mut right_bool = vec![false; cache.sample_size];
        for index in left_set.iter() {
            left_bool[*index] = true;
        }
        for index in right_set.iter() {
            right_bool[*index] = true;
        }

        // If lazy is true, compute the ordered_features from the TrainingCache.
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
            // If lazy is false, compute the ordered_features from the current SubCache
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

        // If lazy is true, compute the ordered_residual from the TrainingCache
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
            // If lazy is false, compute the ordered_residual from the current SubCache
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
        // return result
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
}

/// Calculate the prediction for each leaf node.
/// data: the samples in current node
/// loss: loss type
/// cache: TrainingCache
/// sub_cache: SubCache
#[cfg(feature = "enable_training")]
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
#[cfg(feature = "enable_training")]
fn average(data: &[usize], cache: &TrainingCache) -> ValueType {
    let mut sum: f64 = 0.0;
    let mut weight: f64 = 0.0;

    for index in data.iter() {
        let cv: &CacheValue = &cache.cache_value[*index];
        sum += cv.s;
        weight += cv.c;
    }
    if weight.abs() < 1e-10 {
        0.0
    } else {
        (sum / weight) as ValueType
    }
}

/// The leaf prediction value for LogLikelyhood loss.
#[cfg(feature = "enable_training")]
fn logit_optimal_value(data: &[usize], cache: &TrainingCache) -> ValueType {
    let mut s: f64 = 0.0;
    let mut c: f64 = 0.0;

    for index in data.iter() {
        s += cache.cache_value[*index].s;
        c += f64::from(cache.logit_c[*index]);
    }

    if c.abs() < 1e-10 {
        0.0
    } else {
        (s / c) as ValueType
    }
}

/// The leaf prediction value for LAD loss.
#[cfg(feature = "enable_training")]
fn lad_optimal_value(data: &[usize], cache: &TrainingCache, sub_cache: &SubCache) -> ValueType {
    let sorted_data = cache.sort_with_cache(0, true, data, sub_cache);

    let all_weight = sorted_data
        .iter()
        .fold(0.0f64, |acc, x| acc + cache.cache_value[x.0].c);

    let mut weighted_median: f64 = 0.0;
    let mut weight: f64 = 0.0;
    for (i, pair) in sorted_data.iter().enumerate() {
        weight += cache.cache_value[pair.0].c;
        if (weight * 2.0) > all_weight {
            if i >= 1 {
                weighted_median = f64::from((pair.1 + sorted_data[i - 1].1) / 2.0);
            } else {
                weighted_median = f64::from(pair.1);
            }

            break;
        }
    }
    weighted_median as ValueType
}

/// Return whether the data vector have same target values.
#[allow(unused)]
#[cfg(feature = "enable_training")]
fn same(iv: &[usize], cache: &TrainingCache) -> bool {
    if iv.is_empty() {
        return false;
    }

    let t: ValueType = cache.cache_target[iv[0]];
    for i in iv.iter().skip(1) {
        if !(almost_equal(t, cache.cache_target[*i])) {
            return false;
        }
    }
    true
}

/// The internal node of the decision tree. It's stored in the `value` of the gbdt::binary_tree::BinaryTreeNode
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

    /// Use the `subset` of the `train_data` to train a decision tree
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
    /// let data1 = Data::new_training_data(
    ///     vec![1.0, 2.0, 3.0],
    ///     1.0,
    ///     2.0,
    ///     None
    /// );
    /// let data2 = Data::new_training_data(
    ///     vec![1.1, 2.1, 3.1],
    ///     1.0,
    ///     1.0,
    ///     None
    /// );
    /// let data3 = Data::new_training_data(
    ///     vec![2.0, 2.0, 1.0],
    ///     1.0,
    ///     0.5,
    ///     None
    /// );
    /// let data4 = Data::new_training_data(
    ///     vec![2.0, 2.3, 1.2],
    ///     1.0,
    ///     3.0,
    ///     None
    /// );
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
    /// let mut cache = TrainingCache::get_cache(3, &dv, 2);
    /// let subset = [0,1,2];
    /// tree.fit_n(&dv, &subset, &mut cache);
    /// ```
    #[cfg(feature = "enable_training")]
    pub fn fit_n(&mut self, train_data: &DataVec, subset: &[usize], cache: &mut TrainingCache) {
        assert!(
            self.feature_size == cache.feature_size,
            "Decision_tree and TrainingCache should have same feature size"
        );

        // Compute the TrainingCache
        cache.init_one_iteration(train_data, &self.loss);

        let root_index = self.tree.add_root(BinaryTreeNode::new(DTNode::new()));

        // Generate the SubCache
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
    /// let data1 = Data::new_training_data(
    ///     vec![1.0, 2.0, 3.0],
    ///     1.0,
    ///     1.0,
    ///     None
    /// );
    /// let data2 = Data::new_training_data(
    ///     vec![1.1, 2.1, 3.1],
    ///     1.0,
    ///     1.0,
    ///     None
    /// );
    /// let data3 = Data::new_training_data(
    ///     vec![2.0, 2.0, 1.0],
    ///     1.0,
    ///     2.0,
    ///     None
    /// );
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
    /// let mut cache = TrainingCache::get_cache(3, &dv, 2);
    /// tree.fit(&dv, &mut cache);
    ///
    /// ```
    #[cfg(feature = "enable_training")]
    pub fn fit(&mut self, train_data: &DataVec, cache: &mut TrainingCache) {
        //let mut gain: Vec<ValueType> = vec![0.0; self.feature_size];
        assert!(
            self.feature_size == cache.feature_size,
            "Decision_tree and TrainingCache should have same feature size"
        );
        let data_collection: Vec<usize> = (0..train_data.len()).collect();

        // Compute the TrainingCache
        cache.init_one_iteration(train_data, &self.loss);

        let root_index = self.tree.add_root(BinaryTreeNode::new(DTNode::new()));

        // Generate the SubCache
        let sub_cache = SubCache::get_cache_from_training_cache(cache, train_data, &self.loss);
        self.fit_node(root_index, 0, &data_collection, cache, sub_cache);
    }

    /// Recursively build the tree nodes. It choose a feature and a value to split the node and the data.
    /// And then use the splited data to build the child nodes.
    /// node: the tree index of the current node
    /// depth: the deepth of the current node
    /// train_data: sample data in current node
    /// cache: TrainingCache
    /// sub_cache: SubCache
    #[cfg(feature = "enable_training")]
    fn fit_node(
        &mut self,
        node: TreeIndex,
        depth: u32,
        train_data: &[usize],
        cache: &mut TrainingCache,
        sub_cache: SubCache,
    ) {
        // If the node doesn't need to be splited, make this node a leaf node.
        {
            let node_ref = self
                .tree
                .get_node_mut(node)
                .expect("node should not be empty!");
            // compute the prediction to support unknown features
            node_ref.value.pred = calculate_pred(train_data, &self.loss, cache, &sub_cache);
            if (depth >= self.max_depth)
                || same(train_data, cache)
                || (train_data.len() <= self.min_leaf_size)
            {
                node_ref.value.is_leaf = true;
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
            // If spliting the node is failed, make this node a leaf node
            if splited_data.is_none() {
                node_ref.value.is_leaf = true;
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

    /// Inference the subset of the `test_data`. Return a vector of
    /// predicted values. If the `i` is in the subset, then output[i] is the prediction.
    /// If `i` is not in the subset, then output[i] is 0.0
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
    /// let data1 = Data::new_training_data(
    ///     vec![1.0, 2.0, 3.0],
    ///     1.0,
    ///     2.0,
    ///     None
    /// );
    /// let data2 = Data::new_training_data(
    ///     vec![1.1, 2.1, 3.1],
    ///     1.0,
    ///     1.0,
    ///     None
    /// );
    /// let data3 = Data::new_training_data(
    ///     vec![2.0, 2.0, 1.0],
    ///     1.0,
    ///     0.5,
    ///     None
    /// );
    /// let data4 = Data::new_training_data(
    ///     vec![2.0, 2.3, 1.2],
    ///     1.0,
    ///     3.0,
    ///     None
    /// );
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
    /// let mut cache = TrainingCache::get_cache(3, &dv, 2);
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
    /// // [2.0, 0.75, 0.75, 0.0]
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
    /// let data1 = Data::new_training_data(
    ///     vec![1.0, 2.0, 3.0],
    ///     1.0,
    ///     2.0,
    ///     None
    /// );
    /// let data2 = Data::new_training_data(
    ///     vec![1.1, 2.1, 3.1],
    ///     1.0,
    ///     1.0,
    ///     None
    /// );
    /// let data3 = Data::new_training_data(
    ///     vec![2.0, 2.0, 1.0],
    ///     1.0,
    ///     0.5,
    ///     None
    /// );
    /// let data4 = Data::new_training_data(
    ///     vec![2.0, 2.3, 1.2],
    ///     1.0,
    ///     3.0,
    ///     None
    /// );
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
    /// let mut cache = TrainingCache::get_cache(3, &dv, 2);
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
    ///
    /// Output: (left set, right set, unknown set), feature index, feature value
    #[cfg(feature = "enable_training")]
    fn split(
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
        let mut best_fitness: f64 = std::f64::MAX;

        let mut index: usize = 0;
        let mut value: ValueType = 0.0;

        // Generate the ImpurityCache
        let mut impurity_cache = ImpurityCache::new(cache.sample_size, train_data);

        let mut find: bool = false;
        let mut data_to_split: Vec<(usize, ValueType)> = Vec::new();
        // Calculate each feature's impurity
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
            }
        }

        // Split the node according to the impurity
        if find {
            let mut left: Vec<usize> = Vec::new();
            let mut right: Vec<usize> = Vec::new();
            let mut unknown: Vec<usize> = Vec::new();
            for pair in data_to_split.iter() {
                let (item_index, feature_value) = *pair;
                if feature_value == VALUE_TYPE_UNKNOWN {
                    unknown.push(item_index);
                } else if feature_value < value {
                    left.push(item_index);
                } else {
                    right.push(item_index);
                }
            }
            let mut count: u8 = 0;
            if left.is_empty() {
                count += 1;
            }
            if right.is_empty() {
                count += 1;
            }
            if unknown.is_empty() {
                count += 1;
            }
            if count >= 2 {
                (None, 0, 0.0)
            } else {
                (Some((left, right, unknown)), index, value)
            }
        } else {
            (None, 0, 0.0)
        }
    }

    /// Calculate the impurity.
    /// train_data: samples in current node
    /// feature_index: the index of the selected feature
    /// value: the result of the feature value
    /// impurity: the result of the impurity
    /// cache: TrainingCache
    /// impurity_cache: ImpurityCache
    /// sub_cache: SubCache
    /// output: The sorted data according to the feature
    #[cfg(feature = "enable_training")]
    fn get_impurity(
        train_data: &[usize],
        feature_index: usize,
        value: &mut ValueType,
        impurity: &mut f64,
        cache: &TrainingCache,
        impurity_cache: &mut ImpurityCache,
        sub_cache: &SubCache,
    ) -> Vec<(usize, ValueType)> {
        *impurity = std::f64::MAX;
        *value = VALUE_TYPE_UNKNOWN;
        // Sort the samples with the feature value
        let sorted_data = cache.sort_with_bool_vec(
            feature_index,
            false,
            &impurity_cache.bool_vec,
            train_data.len(),
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
                s += cv.s;
                ss += cv.ss;
                c += cv.c;
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
                impurity_cache.sum_s += cv.s;
                impurity_cache.sum_ss += cv.ss;
                impurity_cache.sum_c += cv.c;
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
            s = cv.s;
            ss = cv.ss;
            c = cv.c;

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
    /// let data1 = Data::new_training_data(
    ///     vec![1.0, 2.0, 3.0],
    ///     1.0,
    ///     2.0,
    ///     None
    /// );
    /// let data2 = Data::new_training_data(
    ///     vec![1.1, 2.1, 3.1],
    ///     1.0,
    ///     1.0,
    ///     None
    /// );
    /// let data3 = Data::new_training_data(
    ///     vec![2.0, 2.0, 1.0],
    ///     1.0,
    ///     0.5,
    ///     None
    /// );
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
    /// let mut cache = TrainingCache::get_cache(3, &dv, 2);
    /// let subset = [0, 1];
    /// tree.fit_n(&dv, &subset, &mut cache);
    ///
    ///
    /// tree.print();
    ///
    /// // output:
    ///
    /// //  ----DTNode { feature_index: 0, feature_value: 1.05, pred: 1.5, is_leaf: false }
    /// //      ----DTNode { feature_index: 0, feature_value: 0.0, pred: 2.0, is_leaf: true }
    /// //      ----DTNode { feature_index: 0, feature_value: 0.0, pred: 1.0, is_leaf: true }
    /// ```
    pub fn print(&self) {
        self.tree.print();
    }

    /// Build a decision tree from xgboost's model. xgboost can dump the model in JSON format. We used serde_json to parse a JSON string.  
    /// # Example
    /// ``` rust
    /// use serde_json::{Result, Value};
    /// use gbdt::decision_tree::DecisionTree;
    /// let data = r#"
    ///       { "nodeid": 0, "depth": 0, "split": 0, "split_condition": 750, "yes": 1, "no": 2, "missing": 2, "children": [
    ///          { "nodeid": 1, "leaf": 25.7333336 },
    ///          { "nodeid": 2, "leaf": 15.791667 }]}"#;
    /// let node: Value = serde_json::from_str(data).unwrap();
    /// let dt = DecisionTree::get_from_xgboost(&node);
    /// ```
    pub fn get_from_xgboost(
        node: &serde_json::Value,
    ) -> Result<Self, Box<dyn Error + Sync + Send>> {
        // Parameters are not used in prediction process, so we use default parameters.
        let mut tree = DecisionTree::new();
        let index = tree.tree.add_root(BinaryTreeNode::new(DTNode::new()));
        tree.add_node_from_json(index, node)?;
        Ok(tree)
    }

    /// Recursively build the tree node from the JSON value.
    fn add_node_from_json(
        &mut self,
        index: TreeIndex,
        node: &serde_json::Value,
    ) -> Result<(), Box<dyn Error + Sync + Send>> {
        {
            let node_ref = self
                .tree
                .get_node_mut(index)
                .expect("node should not be empty!");
            // This is the leaf node
            if let serde_json::Value::Number(pred) = &node["leaf"] {
                let leaf_value = pred.as_f64().ok_or("parse 'leaf' error")?;
                node_ref.value.pred = leaf_value as ValueType;
                node_ref.value.is_leaf = true;
                return Ok(());
            } else {
                // feature value
                let feature_value = node["split_condition"]
                    .as_f64()
                    .ok_or("parse 'split condition' error")?;
                node_ref.value.feature_value = feature_value as ValueType;

                // feature index
                let feature_index = match node["split"].as_i64() {
                    Some(v) => v,
                    None => {
                        let feature_name = node["split"].as_str().ok_or("parse 'split' error")?;
                        let feature_str: String = feature_name.chars().skip(3).collect();
                        feature_str.parse::<i64>()?
                    }
                };
                node_ref.value.feature_index = feature_index as usize;

                // handle unknown feature
                let missing = node["missing"].as_i64().ok_or("parse 'missing' error")?;
                let left_child = node["yes"].as_i64().ok_or("parse 'yes' error")?;
                let right_child = node["no"].as_i64().ok_or("parse 'no' error")?;
                if missing == left_child {
                    node_ref.value.missing = -1;
                } else if missing == right_child {
                    node_ref.value.missing = 1;
                } else {
                    let err: Box<dyn Error> =
                        From::from("not support extra missing node".to_string());
                    return Err(err);
                }
            }
        }

        // ids for children
        let left_child = node["yes"].as_i64().ok_or("parse 'yes' error")?;
        let right_child = node["no"].as_i64().ok_or("parse 'no' error")?;
        let children = node["children"]
            .as_array()
            .ok_or("parse 'children' error")?;
        let mut find_left = false;
        let mut find_right = false;
        for child in children.iter() {
            let node_id = child["nodeid"].as_i64().ok_or("parse 'nodeid' error")?;

            // build left child
            if node_id == left_child {
                find_left = true;
                let left_index = self
                    .tree
                    .add_left_node(index, BinaryTreeNode::new(DTNode::new()));
                self.add_node_from_json(left_index, child)?;
            }

            // build right child
            if node_id == right_child {
                find_right = true;
                let right_index = self
                    .tree
                    .add_right_node(index, BinaryTreeNode::new(DTNode::new()));
                self.add_node_from_json(right_index, child)?;
            }
        }

        if (!find_left) || (!find_right) {
            let err: Box<dyn Error> = From::from("children not found".to_string());
            return Err(err);
        }
        Ok(())
    }

    /// For debug use. Return the number of nodes in current decision tree
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree, TrainingCache};
    /// // set up training data
    /// let data1 = Data::new_training_data(
    ///     vec![1.0, 2.0, 3.0],
    ///     1.0,
    ///     2.0,
    ///     None
    /// );
    /// let data2 = Data::new_training_data(
    ///     vec![1.1, 2.1, 3.1],
    ///     1.0,
    ///     1.0,
    ///     None
    /// );
    /// let data3 = Data::new_training_data(
    ///     vec![2.0, 2.0, 1.0],
    ///     1.0,
    ///     0.5,
    ///     None
    /// );
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
    /// let mut cache = TrainingCache::get_cache(3, &dv, 2);
    /// let subset = [0, 1];
    /// tree.fit_n(&dv, &subset, &mut cache);
    ///
    /// assert_eq!(tree.len(), 3)
    pub fn len(&self) -> usize {
        self.tree.len()
    }

    /// Returns true if the current decision tree is empty
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }
}
