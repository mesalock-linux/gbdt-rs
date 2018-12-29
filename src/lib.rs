//Copyright (C) 2017-2018 Baidu, Inc. All Rights Reserved.
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions
//are met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in
//   the documentation and/or other materials provided with the
//   distribution.
//
// * Neither the name of Baidu, Inc., nor the names of its
//   contributors may be used to endorse or promote products derived
//   from this software without specific prior written permission.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
//"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
//LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
//A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
//OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
//SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
//LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
//DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
//THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
//OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

extern crate rand;

pub mod binary_tree;
pub mod config;
pub mod decision_tree;
pub mod fitness;
pub mod gradient_boost;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn walk_tree() {
        use binary_tree::*;
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

    #[test]
    fn decision_tree() {
        use decision_tree::*;
        let mut tree = DecisionTree::new();
        tree.set_feature_size(3);
        tree.set_max_depth(4);
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

        tree.fit(&dv);

        tree.print();

        let mut dv = Vec::new();
        dv.push(data1.clone());
        dv.push(data2.clone());
        dv.push(data3.clone());
        dv.push(data4.clone());

        println!("{:?}", tree.predict(&dv));
    }

    #[test]
    fn build_decision_tree() {
        use decision_tree::DecisionTree;
        let _ = DecisionTree::new();
    }

    #[test]
    fn config_express() {
        use config::Config;
        let c = Config::empty_config();
        println!("{}", c.to_string());
    }

    #[test]
    fn loss_type() {
        use config::{loss2string, string2loss, LOSS};
        assert_eq!(string2loss("SquaredError"), LOSS::SquaredError);
        assert_eq!(string2loss("LogLikelyhood"), LOSS::LogLikelyhood);
        assert_eq!(string2loss("LAD"), LOSS::LAD);
        assert_eq!(string2loss("WHAT_THE_HUG"), LOSS::UnknownLoss);

        assert_eq!(loss2string(&LOSS::SquaredError), "SquaredError");
        assert_eq!(loss2string(&LOSS::LogLikelyhood), "LogLikelyhood");
        assert_eq!(loss2string(&LOSS::LAD), "LAD");
        assert_eq!(loss2string(&LOSS::UnknownLoss), "UnknownLoss");
    }

    #[test]
    fn fitness() {
        use decision_tree::*;
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

        use fitness::{
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
}
