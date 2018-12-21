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


pub mod binary_tree;
pub mod decision_tree;


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

     #[test]
    fn walk_tree() {
        use super::binary_tree::*;
        let mut tree: BinaryTree<f32> = BinaryTree::new();
        let root = BinaryTreeNode::new(10.0);

        let root_index =tree.add_root(root);

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
    fn build_decision_tree() {
        use super::decision_tree::DecisionTree;
        let tree = DecisionTree::new();
    }

}
