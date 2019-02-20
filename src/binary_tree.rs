//! This module implements a simple binary tree.
//!
//! The tree supports inserting and retriving elements.
//! Deleteing an element is not supported.
//!
//! [std::vec::Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html) is used to implement the binary tree.

/// Node of the binary tree.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct BinaryTreeNode<T> {
    /// Store information in a node.
    pub value: T,

    /// the index of the std::vec::Vec that implements the tree.
    ///
    /// [std::vec::Vec](https://doc.rust-lang.org/std/vec/struct.Vec.html)
    index: usize,

    /// The index of the left child node. 0 means no left child.
    left: usize, // bigger than 0

    /// The index of the right child node. 0 means of right child.
    right: usize, // bigger than 0
}

impl<T> BinaryTreeNode<T> {
    /// New a node with given value
    ///
    /// # Example
    /// ```
    /// use gbdt::binary_tree::BinaryTreeNode;
    /// let root = BinaryTreeNode::new(10.0);
    /// println!("{}", root.value);
    /// ```
    pub fn new(value: T) -> Self {
        BinaryTreeNode {
            value,
            index: 0,
            left: 0,
            right: 0,
        }
    }
}

/// The index to retrive the tree node. Always get the index value from [`BinaryTree`] APIs.
/// Don't directly assign a value to an index.
///
/// [`BinaryTree`]: struct.BinaryTree.html
pub type TreeIndex = usize;

/// The binary tree.
#[derive(Debug, Serialize, Deserialize)]
pub struct BinaryTree<T> {
    tree: Vec<BinaryTreeNode<T>>,
}

impl<T> Default for BinaryTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> BinaryTree<T> {
    /// build a new empty binary tree
    pub fn new() -> Self {
        BinaryTree { tree: Vec::new() }
    }

    /// Add a node as the root node. Return the index of the root node.
    /// # Example
    ///
    /// ``` rust
    /// use gbdt::binary_tree::{BinaryTree, BinaryTreeNode};
    /// let mut tree: BinaryTree<f32> = BinaryTree::new();
    /// let root = BinaryTreeNode::new(10.0);
    /// let root_index = tree.add_root(root);
    /// assert_eq!(root_index, tree.get_root_index());
    /// println!("{}", root_index)
    /// ```
    pub fn add_root(&mut self, root: BinaryTreeNode<T>) -> TreeIndex {
        self.add_node(0, false, root)
    }

    /// Return the index of the root node.
    /// Call this API after inserting root node.
    /// # Example
    ///
    /// ``` rust
    /// use gbdt::binary_tree::{BinaryTree, BinaryTreeNode};
    /// let mut tree: BinaryTree<f32> = BinaryTree::new();
    /// let root = BinaryTreeNode::new(10.0);
    /// let root_index = tree.add_root(root);
    /// assert_eq!(root_index, tree.get_root_index());
    /// println!("{}", root_index)
    /// ```
    pub fn get_root_index(&self) -> TreeIndex {
        0
    }

    /// Return the left child of the given `node`
    /// # Example
    ///
    /// ``` rust
    /// use gbdt::binary_tree::{BinaryTree, BinaryTreeNode};
    /// let mut tree: BinaryTree<f32> = BinaryTree::new();
    /// let root = BinaryTreeNode::new(10.0);
    /// let root_index = tree.add_root(root);
    /// let left_node = BinaryTreeNode::new(5.0);
    /// let _ = tree.add_left_node(root_index, left_node);
    /// let root = tree.get_node(root_index).expect("Didn't find root node");
    /// let left_node = tree.get_left_child(root).expect("Didn't find left child");
    /// println!("{}", left_node.value);
    /// ```
    pub fn get_left_child(&self, node: &BinaryTreeNode<T>) -> Option<&BinaryTreeNode<T>> {
        if node.left == 0 {
            None
        } else {
            self.tree.get(node.left)
        }
    }

    /// Return the right child of the given `node`
    /// # Example
    ///
    /// ``` rust
    /// use gbdt::binary_tree::{BinaryTree, BinaryTreeNode};
    /// let mut tree: BinaryTree<f32> = BinaryTree::new();
    /// let root = BinaryTreeNode::new(10.0);
    /// let root_index = tree.add_root(root);
    /// let right_node = BinaryTreeNode::new(5.0);
    /// let _ = tree.add_right_node(root_index, right_node);
    /// let root = tree.get_node(root_index).expect("Didn't find root node");
    /// let right_node = tree.get_right_child(root).expect("Didn't find right child");
    /// println!("{}", right_node.value);
    /// ```
    pub fn get_right_child(&self, node: &BinaryTreeNode<T>) -> Option<&BinaryTreeNode<T>> {
        if node.right == 0 {
            None
        } else {
            self.tree.get(node.right)
        }
    }

    /// Return the node with the given index
    /// # Example
    ///
    /// ``` rust
    /// use gbdt::binary_tree::{BinaryTree, BinaryTreeNode};
    /// let mut tree: BinaryTree<i32> = BinaryTree::new();
    /// let root = BinaryTreeNode::new(10);
    /// let _ = tree.add_root(root);
    /// let root_index = tree.get_root_index();
    /// let root = tree.get_node(root_index).expect("Didn't find root node");
    /// assert_eq!(10, root.value);
    /// ```
    pub fn get_node(&self, index: TreeIndex) -> Option<&BinaryTreeNode<T>> {
        self.tree.get(index)
    }

    /// Return the muttable reference of a node with the given index
    /// # Example
    ///
    /// ``` rust
    /// use gbdt::binary_tree::{BinaryTree, BinaryTreeNode};
    /// let mut tree: BinaryTree<i32> = BinaryTree::new();
    /// let root = BinaryTreeNode::new(10);
    /// let _ = tree.add_root(root);
    /// let root_index = tree.get_root_index();
    /// let root = tree.get_node_mut(root_index).expect("Didn't find root node");
    /// root.value = 11;
    /// assert_eq!(11, root.value);
    /// ```
    pub fn get_node_mut(&mut self, index: TreeIndex) -> Option<&mut BinaryTreeNode<T>> {
        self.tree.get_mut(index)
    }

    /// Add a node as the left child of a given `parent` node. Return the index of the added node.
    /// # Example
    ///
    /// ``` rust
    /// use gbdt::binary_tree::{BinaryTree, BinaryTreeNode};
    /// let mut tree: BinaryTree<f32> = BinaryTree::new();
    /// let root = BinaryTreeNode::new(10.0);
    /// let root_index = tree.add_root(root);
    /// let left_node = BinaryTreeNode::new(5.0);
    /// let _ = tree.add_left_node(root_index, left_node);
    /// let root = tree.get_node(root_index).expect("Didn't find root node");
    /// let left_node = tree.get_left_child(root).expect("Didn't find left child");
    /// println!("{}", left_node.value);
    /// ```
    pub fn add_left_node(&mut self, parent: TreeIndex, child: BinaryTreeNode<T>) -> TreeIndex {
        self.add_node(parent, true, child)
    }

    /// Add a node as the right child of a given `parent` node. Return the index of the added node.
    /// # Example
    ///
    /// ``` rust
    /// use gbdt::binary_tree::{BinaryTree, BinaryTreeNode};
    /// let mut tree: BinaryTree<f32> = BinaryTree::new();
    /// let root = BinaryTreeNode::new(10.0);
    /// let root_index = tree.add_root(root);
    /// let right_node = BinaryTreeNode::new(5.0);
    /// let _ = tree.add_right_node(root_index, right_node);
    /// let root = tree.get_node(root_index).expect("Didn't find root node");
    /// let right_node = tree.get_right_child(root).expect("Didn't find right child");
    /// println!("{}", right_node.value);
    /// ```
    pub fn add_right_node(&mut self, parent: TreeIndex, child: BinaryTreeNode<T>) -> TreeIndex {
        self.add_node(parent, false, child)
    }

    /// The implmentation of adding a node to the tree. Return the index of the added node
    /// ``parent`` is the index of the parent node. For adding root node, `parent` can be arbitrary
    /// number. When the tree is empty, the node will be added as the root node.
    ///
    /// ``is_left`` means whether the node should be added as left child (true) or right child
    /// (false)
    ///
    /// ``child`` is the node to be added.
    fn add_node(
        &mut self,
        parent: TreeIndex,
        is_left: bool,
        mut child: BinaryTreeNode<T>,
    ) -> TreeIndex {
        child.index = self.tree.len();
        self.tree.push(child);
        let position = self.tree.len() - 1;

        if position == 0 {
            return position;
        }
        if let Some(n) = self.tree.get_mut(parent) {
            if is_left {
                n.left = position;
            } else {
                n.right = position;
            }
        };
        position
    }

    /// For debug use. This API will print the whole tree.
    /// # Example
    /// ```
    /// use gbdt::binary_tree::{BinaryTree, BinaryTreeNode};
    /// let mut tree: BinaryTree<f32> = BinaryTree::new();
    /// let root = BinaryTreeNode::new(10.0);
    ///
    /// let root_index = tree.add_root(root);
    ///
    /// let n1 = BinaryTreeNode::new(5.0);
    /// let n2 = BinaryTreeNode::new(6.0);
    ///
    /// let n1_index = tree.add_left_node(root_index, n1);
    /// let n2_index = tree.add_right_node(root_index, n2);
    ///
    /// let n3 = BinaryTreeNode::new(7.0);
    /// let n4 = BinaryTreeNode::new(8.0);
    ///
    /// tree.add_left_node(n2_index, n3);
    /// tree.add_right_node(n2_index, n4);
    ///
    /// let n5 = BinaryTreeNode::new(9.0);
    ///
    /// tree.add_left_node(n1_index, n5);
    ///
    /// tree.print();
    ///
    /// // Output:
    /// //----10.0
    /// //    ----5.0
    /// //        ----9.0
    /// //    ----6.0
    /// //        ----7.0
    /// //        ----8.0
    ///  ```
    pub fn print(&self)
    where
        T: std::fmt::Debug,
    {
        let mut stack: Vec<(usize, Option<&BinaryTreeNode<T>>)> = Vec::new();
        let root = self.get_node(self.get_root_index());
        stack.push((0, root));
        while !stack.is_empty() {
            let next = stack.pop();
            if let Some((deep, node_opt)) = next {
                if let Some(node) = node_opt {
                    for _i in 0..deep {
                        print!("    ");
                    }
                    println!("----{:?}", node.value);

                    stack.push((deep + 1, self.get_right_child(node)));
                    stack.push((deep + 1, self.get_left_child(node)));
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        self.tree.len()
    }
}
