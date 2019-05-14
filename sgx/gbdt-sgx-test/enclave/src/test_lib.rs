use std::prelude::v1::*;

//#[test]
pub fn walk_tree() {
    use gbdt_sgx::binary_tree::*;
    let mut tree: BinaryTree<f32> = BinaryTree::new();
    let root = BinaryTreeNode::new(10.0);

    let root_index = tree.add_root(root);

    let n1 = BinaryTreeNode::new(5.0);
    let n2 = BinaryTreeNode::new(6.0);

    let n1_index = tree.add_left_node(root_index, n1);
    let n2_index = tree.add_right_node(root_index, n2);

    let n3 = BinaryTreeNode::new(7.0);
    let n4 = BinaryTreeNode::new(8.0);

    tree.add_left_node(n2_index, n3);
    tree.add_right_node(n2_index, n4);

    let n5 = BinaryTreeNode::new(9.0);

    tree.add_left_node(n1_index, n5);

    tree.print();
}

//#[test]
pub fn decision_tree() {
    use gbdt_sgx::config::Loss;
    use gbdt_sgx::decision_tree::*;
    let mut tree = DecisionTree::new();
    tree.set_feature_size(3);
    tree.set_max_depth(2);
    tree.set_min_leaf_size(1);
    tree.set_loss(Loss::SquaredError);
    let data1 = Data {
        feature: vec![1.0, 2.0, 3.0],
        target: 2.0,
        weight: 1.0,
        label: 1.0,
        residual: 1.0,
        initial_guess: 1.0,
    };
    let data2 = Data {
        feature: vec![1.1, 2.1, 3.1],
        target: 1.0,
        weight: 1.0,
        label: 1.0,
        residual: 1.0,
        initial_guess: 1.0,
    };
    let data3 = Data {
        feature: vec![2.0, 2.0, 1.0],
        target: 0.5,
        weight: 1.0,
        label: 2.0,
        residual: 2.0,
        initial_guess: 2.0,
    };
    let data4 = Data {
        feature: vec![2.0, 2.3, 1.2],
        target: 3.0,
        weight: 1.0,
        label: 0.0,
        residual: 0.0,
        initial_guess: 1.0,
    };

    let mut dv = Vec::new();
    dv.push(data1.clone());
    dv.push(data2.clone());
    dv.push(data3.clone());
    dv.push(data4.clone());

    let mut cache = TrainingCache::get_cache(3, &dv, 3);
    println!("2here");
    tree.fit(&dv, &mut cache);
    println!("3here");

    tree.print();

    let mut dv = Vec::new();
    dv.push(data1.clone());
    dv.push(data2.clone());
    dv.push(data3.clone());
    dv.push(data4.clone());

    println!("{:?}", tree.predict(&dv));
}

//#[test]
pub fn build_decision_tree() {
    use gbdt_sgx::decision_tree::DecisionTree;
    let _ = DecisionTree::new();
}

//#[test]
pub fn config_express() {
    use gbdt_sgx::config::Config;
    let c = Config::new();
    println!("{}", c.to_string());
}

//#[test]
pub fn loss_type() {
    use gbdt_sgx::config::{loss2string, string2loss, Loss};
    assert_eq!(string2loss("SquaredError"), Loss::SquaredError);
    assert_eq!(string2loss("LogLikelyhood"), Loss::LogLikelyhood);
    assert_eq!(string2loss("LAD"), Loss::LAD);

    assert_eq!(loss2string(&Loss::SquaredError), "SquaredError");
    assert_eq!(loss2string(&Loss::LogLikelyhood), "LogLikelyhood");
    assert_eq!(loss2string(&Loss::LAD), "LAD");
}

//#[test]
pub fn fitness() {
    use gbdt_sgx::decision_tree::*;
    let mut dv: DataVec = Vec::new();
    dv.push(Data {
        feature: Vec::new(),
        target: 1.0,
        weight: 0.1,
        label: 1.0,
        residual: 0.5,
        initial_guess: VALUE_TYPE_UNKNOWN,
    });
    dv.push(Data {
        feature: Vec::new(),
        target: 1.0,
        weight: 0.2,
        label: 0.0,
        residual: 0.5,
        initial_guess: VALUE_TYPE_UNKNOWN,
    });
    dv.push(Data {
        feature: Vec::new(),
        target: 0.0,
        weight: 0.3,
        label: 1.0,
        residual: 0.5,
        initial_guess: VALUE_TYPE_UNKNOWN,
    });
    dv.push(Data {
        feature: Vec::new(),
        target: 0.0,
        weight: 0.4,
        label: 0.0,
        residual: 0.5,
        initial_guess: VALUE_TYPE_UNKNOWN,
    });

    use gbdt_sgx::fitness::{
        almost_equal, average, label_average, same, weighted_label_median,
        weighted_residual_median,
    };
    assert_eq!(true, almost_equal(0.1, 0.100000000001));
    assert_eq!(false, same(&dv, dv.len()));
    assert!(almost_equal(0.3, average(&dv, dv.len())));
    assert!(almost_equal(0.4, label_average(&dv, dv.len())));
    assert!(almost_equal(0.0, weighted_label_median(&dv, dv.len())));
    assert!(almost_equal(0.5, weighted_residual_median(&dv, dv.len())));
}
