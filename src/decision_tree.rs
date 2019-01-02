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
//! use gbdt::decision_tree::{Data, DecisionTree};
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
//! tree.fit(&dv);
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

use rand::prelude::SliceRandom;
use rand::thread_rng;

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
def_value_type!(f64);

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
fn calculate_pred(data: &[&Data], loss: &Loss) -> ValueType {
    match loss {
        Loss::SquaredError => average(data),
        Loss::LogLikelyhood => logit_optimal_value(data),
        Loss::LAD => lad_optimal_value(data),
    }
}

/// The leaf prediction value for SquaredError loss.
fn average(data: &[&Data]) -> ValueType {
    let mut sum: ValueType = 0.0;
    let mut weight: ValueType = 0.0;
    for elem in data.iter() {
        sum += elem.target * elem.weight;
        weight += elem.weight;
    }

    sum / weight
}

/// The leaf prediction value for LogLikelyhood loss.
fn logit_optimal_value(data: &[&Data]) -> ValueType {
    let mut s: ValueType = 0.0;
    let mut c: ValueType = 0.0;

    for elem in data.iter() {
        s += elem.target * elem.weight;
        let y = elem.target.abs();
        c += y * (2.0 - y) * elem.weight;
    }

    if (c - 0.0).abs() < 1e-10 {
        0.0
    } else {
        s / c
    }
}

/// The leaf prediction value for LAD loss.
fn lad_optimal_value(data: &[&Data]) -> ValueType {
    let mut data_copy = data.to_vec();
    data_copy.sort_by(|a, b| {
        let v1: ValueType = a.residual;
        let v2: ValueType = b.residual;
        v1.partial_cmp(&v2).unwrap()
    });

    let mut all_weight: ValueType = 0.0;
    for elem in data_copy.iter() {
        all_weight += elem.weight;
    }

    let mut weighted_median: ValueType = 0.0;
    let mut weight = 0.0;
    for i in 0..data_copy.len() {
        weight += data_copy[i].weight;
        if (weight * 2.0) > all_weight {
            if i >= 1 {
                weighted_median = (data_copy[i].residual + data_copy[i - 1].residual) / 2.0;
            } else {
                weighted_median = data_copy[i].residual;
            }
            break;
        }
    }

    weighted_median
}

/// Return whether the data vector have same target values.
fn same(dv: &[&Data]) -> bool {
    if dv.is_empty() {
        return false;
    }

    let t: ValueType = dv[0].target;
    for i in dv.iter().skip(1) {
        if !(almost_equal(t, i.target)) {
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
    /// use gbdt::decision_tree::{Data, DecisionTree};
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
    /// tree.fit_n(&dv, 2);
    ///
    /// ```
    pub fn fit_n(&mut self, train_data: &DataVec, n: usize) {
        let data: Vec<&Data> = (0..std::cmp::min(n, train_data.len()))
            .filter_map(|x| train_data.get(x))
            .collect();
        //let mut gain: Vec<ValueType> = vec![0.0; self.feature_size];
        let root_index = self.tree.add_root(BinaryTreeNode::new(DTNode::new()));
        self.fit_node(root_index, 0, &data);
    }

    /// Use the samples in `train_data` to train the decision tree.
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
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
    /// tree.fit(&dv);
    ///
    /// ```
    pub fn fit(&mut self, train_data: &DataVec) {
        let data: Vec<&Data> = (0..train_data.len())
            .filter_map(|x| train_data.get(x))
            .collect();
        //let mut gain: Vec<ValueType> = vec![0.0; self.feature_size];
        let root_index = self.tree.add_root(BinaryTreeNode::new(DTNode::new()));
        self.fit_node(root_index, 0, &data);
    }

    /// Recursively build the tree nodes. It choose a feature and a value to split the node and the data.
    /// And then use the splited data to build the child nodes.
    fn fit_node(&mut self, node: TreeIndex, depth: u32, train_data: &[&Data]) {
        // If the node doesn't need to be splited.
        {
            let node_ref = self
                .tree
                .get_node_mut(node)
                .expect("node should not be empty!");
            if (depth >= self.max_depth)
                || same(train_data)
                || (train_data.len() <= self.min_leaf_size)
            {
                node_ref.value.is_leaf = true;
                node_ref.value.pred = calculate_pred(train_data, &self.loss);
                return;
            }
        }

        // Try to find a feature and a value to split the node.
        let (splited_data, feature_index, feature_value) =
            DecisionTree::split(train_data, self.feature_size, self.feature_sample_ratio);

        {
            let node_ref = self
                .tree
                .get_node_mut(node)
                .expect("node should not be empty");
            if splited_data.is_none() {
                node_ref.value.is_leaf = true;
                node_ref.value.pred = calculate_pred(train_data, &self.loss);
                return;
            } else {
                node_ref.value.feature_index = feature_index;
                node_ref.value.feature_value = feature_value;
            }
        }

        // Use the splited data to build child nodes.
        if let Some((left_data, right_data)) = splited_data {
            let left_index = self
                .tree
                .add_left_node(node, BinaryTreeNode::new(DTNode::new()));
            self.fit_node(left_index, depth + 1, &left_data);
            let right_index = self
                .tree
                .add_right_node(node, BinaryTreeNode::new(DTNode::new()));
            self.fit_node(right_index, depth + 1, &right_data);
        }
    }

    /// Inference the values of the first `n` samples in the `test_data`. Return a vector of
    /// predicted values.
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
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
    /// tree.fit(&dv);
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
    /// println!("{:?}", tree.predict_n(&dv, 3));
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
    pub fn predict_n(&self, test_data: &DataVec, n: usize) -> PredVec {
        let root = self
            .tree
            .get_node(self.tree.get_root_index())
            .expect("Decision tree should have root node");

        // Inference the samples one by one.
        test_data
            .iter()
            .take(std::cmp::min(n, test_data.len()))
            .map(|x| self.predict_one(root, x))
            .collect()
    }

    /// Inference the values of samples in the `test_data`. Return a vector of the predicted
    /// values.
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
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
    /// tree.fit(&dv);
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
        // return the node's prediction
        if node.value.is_leaf {
            node.value.pred
        } else {
            assert!(
                sample.feature.len() > node.value.feature_index,
                "sample doesn't have the feature"
            );
            // choose a child node, call this function again
            if sample.feature[node.value.feature_index] < node.value.feature_value {
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
    fn split<'a>(
        train_data: &'a [&Data],
        feature_size: usize,
        feature_sample_ratio: f64,
    ) -> (Option<(Vec<&'a Data>, Vec<&'a Data>)>, usize, ValueType) {
        let mut fs = feature_size;
        let mut fv: Vec<usize> = (0..).take(fs).collect();

        let mut rng = thread_rng();
        if feature_sample_ratio < 1.0 {
            fs = (feature_sample_ratio * (feature_size as f64)) as usize;
            fv.shuffle(&mut rng);
        }

        let mut v: ValueType = 0.0;
        let mut impurity: ValueType = 0.0;
        //let mut g: f64 = 0.0;
        let mut best_fitness: ValueType = VALUE_TYPE_MAX;

        let mut index: usize = 0;
        let mut value: ValueType = 0.0;
        // let mut gain: f64 = 0.0;

        let mut find: bool = false;
        for i in fv.iter().take(fs) {
            DecisionTree::get_impurity(train_data, *i, &mut v, &mut impurity);
            if best_fitness > impurity {
                find = true;
                best_fitness = impurity;
                index = *i;
                value = v;
                //gain = g;
            }
        }
        if find {
            let mut left: Vec<&Data> = Vec::new();
            let mut right: Vec<&Data> = Vec::new();
            for elem in train_data.iter() {
                if let Some(v) = elem.feature.get(index) {
                    if *v < value {
                        left.push(*elem);
                    } else {
                        right.push(*elem);
                    }
                } else {
                    assert!(true, "feature can't be empty");
                }
            }
            (Some((left, right)), index, value)
        } else {
            (None, 0, 0.0)
        }
    }

    /// Calculate the impurity.
    fn get_impurity(
        train_data: &[&Data],
        feature_index: usize,
        value: &mut ValueType,
        impurity: &mut ValueType,
        //gain: &mut f64,
    ) {
        *impurity = VALUE_TYPE_MAX;
        let mut data = train_data.to_vec();

        for elem in train_data.iter() {
            assert!(
                elem.feature.get(feature_index).is_some(),
                "feature is unknown"
            );
        }

        let index = feature_index;
        data.sort_by(|a, b| {
            let v1: ValueType = a.feature[index];
            let v2: ValueType = b.feature[index];
            v1.partial_cmp(&v2).unwrap()
        });

        let fitness0: ValueType = 0.0;

        let mut s: ValueType = 0.0;
        let mut ss: ValueType = 0.0;
        let mut c: ValueType = 0.0;

        for i in data.iter().take(train_data.len()) {
            s += i.target * i.weight;
            ss += i.target * i.target * i.weight;
            c += i.weight;
        }

        // fitness00 is designed to support unknown feature
        // Supress the warning here by add '_' before it
        // TODO: remove '_' to support unknown feature
        let _fitness00: ValueType = if c > 1.0 { ss - s * s / c } else { 0.0 };

        let mut ls: ValueType = 0.0;
        let mut lss: ValueType = 0.0;
        let mut lc: ValueType = 0.0;
        let mut rs: ValueType = s;
        let mut rss: ValueType = ss;
        let mut rc: ValueType = c;

        for i in 0..(train_data.len() - 1) {
            s = data[i].target * data[i].weight;
            ss = data[i].target * data[i].target * data[i].weight;
            c = data[i].weight;

            ls += s;
            lss += ss;
            lc += c;

            rs -= s;
            rss -= ss;
            rc -= c;

            let f1: ValueType = data[i].feature[index];
            let f2: ValueType = data[i + 1].feature[index];

            if almost_equal(f1, f2) {
                continue;
            }

            let fitness1 = if lc > 1.0 { lss - ls * ls / lc } else { 0.0 };

            let fitness2 = if rc > 1.0 { rss - rs * rs / rc } else { 0.0 };

            let fitness: ValueType = fitness0 + fitness1 + fitness2;

            if *impurity > fitness {
                *impurity = fitness;
                *value = (f1 + f2) / 2.0;
                //*gain = fitness00 - fitness1 - fitness2;
            }
        }
    }

    /// Print the decision tree. For debug use.
    ///
    /// # Example
    /// ```
    /// use gbdt::config::Loss;
    /// use gbdt::decision_tree::{Data, DecisionTree};
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
    /// tree.fit_n(&dv, 2);
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
}
