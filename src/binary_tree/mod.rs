#[derive(Debug)]
pub struct BinaryTreeNode<T> {
    pub value: T,
    index: usize,
    left: usize, // bigger than 0
    right: usize, // bigger than 0
}


impl <T> BinaryTreeNode<T> {
    pub fn new(value: T) -> Self {
        BinaryTreeNode {
            value: value,
            index: 0,
            left: 0,
            right: 0,
        }
    }
}

pub type TreeIndex = usize;

#[derive(Debug)]
pub struct BinaryTree<T> {
    tree: Vec<BinaryTreeNode<T>>,
}

impl <T> BinaryTree<T> {
    pub fn new() -> Self {
        let tree: Vec<BinaryTreeNode<T>> = Vec::new();
        BinaryTree {
            tree: tree,
        }
    }

    pub fn get_root_index(&self) -> TreeIndex {
        0
    }

    pub fn get_left_child(&self, node: &BinaryTreeNode<T>) -> Option<&BinaryTreeNode<T>> {
        if node.left == 0 {
            None
        } else {
            self.tree.get(node.left)
        }
    }

    pub fn get_right_child(&self, node: &BinaryTreeNode<T>) -> Option<&BinaryTreeNode<T>> {
        if node.right == 0 {
            None
        } else {
            self.tree.get(node.right)
        }
    }

    pub fn get_node(&self, index: TreeIndex) -> Option<&BinaryTreeNode<T>> {
        self.tree.get(index)
    }

    pub fn get_node_mut(&mut self, index: TreeIndex) -> Option<&mut BinaryTreeNode<T>> {
        self.tree.get_mut(index)
    }

    pub fn add_root(&mut self, root: BinaryTreeNode<T>) -> TreeIndex {
        self.add_node(0, false, root)
    }
    
    pub fn add_left_node(&mut self, parent: TreeIndex, child: BinaryTreeNode<T>) -> TreeIndex {
        self.add_node(parent, true, child)
    }

    pub fn add_right_node(&mut self, parent: TreeIndex, child: BinaryTreeNode<T>) -> TreeIndex {
        self.add_node(parent, false, child)
    }

    fn add_node(&mut self, parent: TreeIndex, is_left: bool, mut child: BinaryTreeNode<T>) -> TreeIndex {
        child.index = self.tree.len();
        self.tree.push(child);
        let position = self.tree.len()-1;

        if position == 0 {
            return position;
        }
        self.tree.get_mut(parent).map(|n| {
            if is_left {
                n.left = position;
            }
            else {
                n.right = position;
            }
        });
        position
    }


    pub fn print(&self) 
        where T: std::fmt::Debug {
        let mut stack: Vec<(usize, Option<&BinaryTreeNode<T>>)> = Vec::new();
        let root = self.get_node(self.get_root_index());
        stack.push((0, root));
        while stack.len() > 0 {
            let next = stack.pop();
            if let Some((deep, node_opt)) = next {
                if let Some(node) = node_opt {
                    for _i in 0..deep {
                        print!("    ");
                    }
                    println!("----{:?}", node.value);
                    
                    stack.push((deep+1, self.get_right_child(node)));
                    stack.push((deep+1, self.get_left_child(node)));
                }
            }
        }
    }
}
