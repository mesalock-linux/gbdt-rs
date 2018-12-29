use binary_tree::BinaryTree;
use binary_tree::BinaryTreeNode;
use binary_tree::TreeIndex;
use fitness::almost_equal;

use rand::prelude::SliceRandom;
use rand::thread_rng;

// use continous variables for decision tree
pub type ValueType = f64;

#[derive(Debug, Clone)]
pub struct Data {
    pub feature: Vec<ValueType>,
    pub target: ValueType,
    pub weight: ValueType,
    pub label: ValueType,
    pub residual: ValueType,
    pub initial_guess: ValueType,
}

pub type DataVec = Vec<Data>;
pub type PredVec = Vec<ValueType>;

pub const VALUE_TYPE_MAX: f64 = std::f64::MAX;
pub const VALUE_TYPE_MIN: f64 = std::f64::MIN;
pub const VALUE_TYPE_UNKNOWN: f64 = VALUE_TYPE_MIN;

pub enum Loss {
    SquaredError,
    LogLikelyhood,
    LAD,
}

fn calculate_pred(data: &[&Data], loss: &Loss) -> ValueType {
    match loss {
        Loss::SquaredError => average(data),
        Loss::LogLikelyhood => logit_optimal_value(data),
        Loss::LAD => lad_optimal_value(data),
    }
}

fn average(data: &[&Data]) -> ValueType {
    let mut sum: ValueType = 0.0;
    let mut weight: ValueType = 0.0;
    for elem in data.iter() {
        sum += elem.target * elem.weight;
        weight += elem.weight;
    }

    sum / weight
}

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

fn lad_optimal_value(data: &[&Data]) -> ValueType {
    let mut data_copy = data.to_vec();
    data_copy.sort_by(|a, b| {
        let v1: ValueType = a.residual;
        let v2: ValueType = b.residual;
        v1.partial_cmp(&v2).unwrap()
    });

    let mut all_weight: f64 = 0.0;
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

#[derive(Debug)]
struct DTNode {
    feature_index: usize,
    feature_value: ValueType,
    pred: ValueType,
    is_leaf: bool,
}

impl DTNode {
    pub fn new() -> Self {
        DTNode {
            feature_index: 0,
            feature_value: 0.0,
            pred: 0.0,
            is_leaf: false,
        }
    }
}

pub struct DecisionTree {
    tree: BinaryTree<DTNode>,
    feature_size: usize,
    max_depth: u32,
    min_leaf_size: usize,
    loss: Loss,
    feature_sample_ratio: f64,
}

impl Default for DecisionTree {
    fn default() -> Self {
        Self::new()
    }
}

impl DecisionTree {
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

    pub fn set_feature_size(&mut self, size: usize) {
        self.feature_size = size;
    }

    pub fn set_max_depth(&mut self, max_depth: u32) {
        self.max_depth = max_depth;
    }

    pub fn set_min_leaf_size(&mut self, min_leaf_size: usize) {
        self.min_leaf_size = min_leaf_size;
    }

    pub fn set_loss(&mut self, loss: Loss) {
        self.loss = loss;
    }

    pub fn set_feature_sample_ratio(&mut self, feature_sample_ratio: f64) {
        self.feature_sample_ratio = feature_sample_ratio;
    }

    pub fn fit_n(&mut self, train_data: &DataVec, n: usize) {
        let sample_size = if n < train_data.len() {
            n
        } else {
            train_data.len()
        };

        let mut data: Vec<&Data> = Vec::new();
        for i in 0..sample_size {
            let ele = train_data.get(i);
            if let Some(sample) = ele {
                data.push(sample);
            }
        }
        //let mut gain: Vec<ValueType> = vec![0.0; self.feature_size];
        let root_index = self.tree.add_root(BinaryTreeNode::new(DTNode::new()));
        self.fit_node(root_index, 0, &data);
    }

    pub fn fit(&mut self, train_data: &DataVec) {
        let mut data: Vec<&Data> = Vec::new();
        for i in 0..train_data.len() {
            let ele = train_data.get(i);
            if let Some(sample) = ele {
                data.push(sample);
            }
        }
        //let mut gain: Vec<ValueType> = vec![0.0; self.feature_size];
        let root_index = self.tree.add_root(BinaryTreeNode::new(DTNode::new()));
        self.fit_node(root_index, 0, &data);
    }

    fn fit_node(&mut self, node: TreeIndex, depth: u32, train_data: &[&Data]) {
        // modify current node
        {
            let node_option = self.tree.get_node_mut(node);
            assert_eq!(node_option.is_some(), true);
            let node_ref = node_option.unwrap();
            if (depth > self.max_depth)
                || same(train_data)
                || (train_data.len() < self.min_leaf_size)
            {
                node_ref.value.is_leaf = true;
                node_ref.value.pred = calculate_pred(train_data, &self.loss);
                return;
            }
        }

        let (splited_data, feature_index, feature_value) =
            DecisionTree::split(train_data, self.feature_size, self.feature_sample_ratio);

        {
            let node_option = self.tree.get_node_mut(node);
            assert_eq!(node_option.is_some(), true);
            let node_ref = node_option.unwrap();
            if splited_data.is_none() {
                node_ref.value.is_leaf = true;
                node_ref.value.pred = calculate_pred(train_data, &self.loss);
                return;
            } else {
                node_ref.value.feature_index = feature_index;
                node_ref.value.feature_value = feature_value;
            }
        }

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

    pub fn predict_n(&self, test_data: &DataVec, n: usize) -> PredVec {
        let sample_size = if n < test_data.len() {
            n
        } else {
            test_data.len()
        };
        let mut ret: PredVec = Vec::new();
        let root = self.tree.get_node(self.tree.get_root_index());
        assert!(root.is_some(), "Decision tree should have root node");
        let root = root.unwrap();
        for i in test_data.iter().take(sample_size) {
            ret.push(self.predict_one(root, &i));
        }
        ret
    }
    pub fn predict(&self, test_data: &DataVec) -> PredVec {
        let mut ret: PredVec = Vec::new();
        let root = self.tree.get_node(self.tree.get_root_index());
        assert!(root.is_some(), "Decision tree should have root node");
        let root = root.unwrap();
        for elem in test_data.iter() {
            ret.push(self.predict_one(root, elem));
        }
        ret
    }

    fn predict_one(&self, node: &BinaryTreeNode<DTNode>, sample: &Data) -> ValueType {
        if node.value.is_leaf {
            node.value.pred
        } else {
            assert!(
                sample.feature.len() > node.value.feature_index,
                "sample doesn't have the feature"
            );
            if sample.feature[node.value.feature_index] < node.value.feature_value {
                let left = self.tree.get_left_child(node);
                assert!(left.is_some(), "Left child shouldn't be None");
                self.predict_one(left.unwrap(), sample)
            } else {
                let right = self.tree.get_right_child(node);
                assert!(right.is_some(), "Right child shouldn't be None");
                self.predict_one(right.unwrap(), sample)
            }
        }
    }

    fn split<'a>(
        train_data: &'a [&Data],
        feature_size: usize,
        feature_sample_ratio: f64,
    ) -> (Option<(Vec<&'a Data>, Vec<&'a Data>)>, usize, ValueType) {
        let mut fs = feature_size;
        let mut fv: Vec<usize> = Vec::new();
        for i in 0..fs {
            fv.push(i);
        }

        let mut rng = thread_rng();
        if feature_sample_ratio < 1.0 {
            fs = (feature_sample_ratio * (feature_size as f64)) as usize;
            fv.shuffle(&mut rng);
        }

        let mut v: ValueType = 0.0;
        let mut impurity: f64 = 0.0;
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

    fn get_impurity(
        train_data: &[&Data],
        feature_index: usize,
        value: &mut ValueType,
        impurity: &mut f64,
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

        let fitness0: f64 = 0.0;

        let mut s: f64 = 0.0;
        let mut ss: f64 = 0.0;
        let mut c: f64 = 0.0;

        for i in data.iter().take(train_data.len()) {
            s += i.target * i.weight;
            ss += i.target * i.target * i.weight;
            c += i.weight;
        }

        // fitness00 is designed to support unknown feature
        // Supress the warning here by add '_' before it
        // TODO: remove '_' to support unknown feature
        let _fitness00: ValueType = if c > 1.0 { ss - s * s / c } else { 0.0 };

        let mut ls: f64 = 0.0;
        let mut lss: f64 = 0.0;
        let mut lc: f64 = 0.0;
        let mut rs: f64 = s;
        let mut rss: f64 = ss;
        let mut rc: f64 = c;

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

            let mut f1: ValueType = data[i].feature[index];
            let mut f2: ValueType = data[i + 1].feature[index];

            if (f1 - f2).abs() < 1.0e-5 {
                continue;
            }

            let fitness1 = if lc > 1.0 { lss - ls * ls / lc } else { 0.0 };

            let fitness2 = if rc > 1.0 { rss - rs * rs / rc } else { 0.0 };

            let mut fitness: ValueType = fitness0 + fitness1 + fitness2;

            if *impurity > fitness {
                *impurity = fitness;
                *value = (f1 + f2) / 2.0;
                //*gain = fitness00 - fitness1 - fitness2;
            }
        }
    }
    pub fn print(&self) {
        self.tree.print();
    }
}
