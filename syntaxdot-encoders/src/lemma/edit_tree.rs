// Copyright 2019 The edit_tree contributors
// Copyright 2020 TensorDot
//
// Licensed under the Apache License, Version 2.0 or the MIT license,
// at your option.
//
// Contributors:
//
// Tobias Pütz <tobias.puetz@uni-tuebingen.de>
// Daniël de Kok <me@danieldk.eu>

//! Edit trees

use std::cmp::Eq;
use std::cmp::Ordering;
use std::fmt::Debug;

use lazy_static::lazy_static;
use seqalign::measures::LCSOp;
use seqalign::measures::LCS;
use seqalign::op::IndexedOperation;
use seqalign::Align;
use serde::{Deserialize, Serialize};

lazy_static! {
    static ref MEASURE: LCS = LCS::new(1, 1);
}

/// Enum representing a `TreeNode` of an `Graph<TreeNode<T>,Place>`.
#[derive(Debug, PartialEq, Hash, Eq, Clone, Serialize, Deserialize)]
pub enum EditTree {
    MatchNode {
        pre: usize,
        suf: usize,
        left: Option<Box<EditTree>>,
        right: Option<Box<EditTree>>,
    },

    ReplaceNode {
        replacee: Vec<char>,
        replacement: Vec<char>,
    },
}

impl EditTree {
    /// Returns a edit tree specifying how to derive `b` from `a`.
    ///
    /// **Caution:** when using with stringy types. UTF-8 multi byte
    /// chars will not be treated well. Consider passing in &[char]
    /// instead.
    pub fn create_tree(a: &[char], b: &[char]) -> Option<Self> {
        build_tree(a, b).map(|tree| *tree)
    }

    /// Recursively applies the nodes stored in the edit tree. Returns `None` if the tree is not applicable to
    /// `form`.
    pub fn apply(&self, form: &[char]) -> Option<Vec<char>> {
        let form_len = form.len();
        match self {
            EditTree::MatchNode {
                pre,
                suf,
                left,
                right,
            } => {
                if pre + suf >= form_len {
                    return None;
                }

                let mut left = match left {
                    Some(left) => left.apply(&form[..*pre])?,
                    None => vec![],
                };

                left.extend(form[*pre..form_len - *suf].iter().cloned());

                if let Some(right) = right {
                    left.extend(right.apply(&form[form_len - *suf..])?)
                }

                Some(left)
            }

            EditTree::ReplaceNode {
                ref replacee,
                ref replacement,
            } => {
                if form == &replacee[..] || replacee.is_empty() {
                    Some(replacement.clone())
                } else {
                    None
                }
            }
        }
    }
}

/// Struct representing a continuous match between two sequences.
#[derive(Debug, PartialEq, Eq, Hash)]
struct LcsMatch {
    start_src: usize,
    start_targ: usize,
    length: usize,
}

impl LcsMatch {
    fn new(start_src: usize, start_targ: usize, length: usize) -> Self {
        LcsMatch {
            start_src,
            start_targ,
            length,
        }
    }
    fn empty() -> Self {
        LcsMatch::new(0, 0, 0)
    }
}

impl PartialOrd for LcsMatch {
    fn partial_cmp(&self, other: &LcsMatch) -> Option<Ordering> {
        Some(self.length.cmp(&other.length))
    }
}

/// Returns the start and end index of the longest match. Returns none if no match is found.
fn longest_match(script: &[IndexedOperation<LCSOp>]) -> Option<LcsMatch> {
    let mut longest = LcsMatch::empty();

    let mut script_slice = script;
    while !script_slice.is_empty() {
        let op = &script_slice[0];

        match op.operation() {
            LCSOp::Match => {
                let in_start = op.source_idx();
                let o_start = op.target_idx();
                let end = match script_slice
                    .iter()
                    .position(|x| !matches!(x.operation(), LCSOp::Match))
                {
                    Some(idx) => idx,
                    None => script_slice.len(),
                };
                if end > longest.length {
                    longest = LcsMatch::new(in_start, o_start, end);
                };

                script_slice = &script_slice[end..];
            }
            _ => {
                script_slice = &script_slice[1..];
            }
        }
    }

    if longest.length != 0 {
        Some(longest)
    } else {
        None
    }
}

/// Recursively builds an edit tree by applying itself to pre and suffix of the longest common substring.
fn build_tree(form_ch: &[char], lem_ch: &[char]) -> Option<Box<EditTree>> {
    if form_ch.is_empty() && lem_ch.is_empty() {
        return None;
    }

    let alignment = MEASURE.align(form_ch, lem_ch);
    let root = match longest_match(&alignment.edit_script()[..]) {
        Some(m) => EditTree::MatchNode {
            pre: m.start_src,
            suf: (form_ch.len() - m.start_src) - m.length,
            left: build_tree(&form_ch[..m.start_src], &lem_ch[..m.start_targ]),
            right: build_tree(
                &form_ch[m.start_src + m.length..],
                &lem_ch[m.start_targ + m.length..],
            ),
        },
        None => EditTree::ReplaceNode {
            replacee: form_ch.to_vec(),
            replacement: lem_ch.to_vec(),
        },
    };
    Some(Box::new(root))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::EditTree;

    /// Utility trait to retrieve a lower-cased `Vec<char>`.
    pub trait ToLowerCharVec {
        fn to_lower_char_vec(&self) -> Vec<char>;
    }

    impl<'a> ToLowerCharVec for &'a str {
        fn to_lower_char_vec(&self) -> Vec<char> {
            self.to_lowercase().chars().collect()
        }
    }

    #[test]
    fn test_graph_equality_outcome() {
        let a = "hates".to_lower_char_vec();
        let b = "hate".to_lower_char_vec();
        let g = EditTree::create_tree(&a, &b).unwrap();

        let a = "loves".to_lower_char_vec();
        let b = "love".to_lower_char_vec();
        let g1 = EditTree::create_tree(&a, &b).unwrap();

        let f = "loves".to_lower_char_vec();
        let f1 = "hates".to_lower_char_vec();
        let exp = "love".to_lower_char_vec();
        let exp1 = "hate".to_lower_char_vec();

        assert_eq!(g.apply(&f1).unwrap(), exp1);
        assert_eq!(g1.apply(&f).unwrap(), exp);
        assert_eq!(g, g1);
    }

    #[test]
    fn test_graph_equality_outcome_2() {
        let g = EditTree::create_tree(
            &"machen".to_lower_char_vec(),
            &"gemacht".to_lower_char_vec(),
        )
        .unwrap();
        let g1 = EditTree::create_tree(
            &"lachen".to_lower_char_vec(),
            &"gelacht".to_lower_char_vec(),
        )
        .unwrap();

        let f = "machen".to_lower_char_vec();
        let f1 = "lachen".to_lower_char_vec();
        let exp = "gemacht".to_lower_char_vec();
        let exp1 = "gelacht".to_lower_char_vec();

        assert_eq!(g.apply(&f1).unwrap(), exp1);
        assert_eq!(g1.apply(&f).unwrap(), exp);
        assert_eq!(g, g1);
    }

    #[test]
    fn test_graph_equality_outcome_3() {
        let a = "aaaaaaaaen".to_lower_char_vec();
        let b = "geaaaaaaaat".to_lower_char_vec();
        let g = EditTree::create_tree(&a, &b).unwrap();

        let a = "lachen".to_lower_char_vec();
        let b = "gelacht".to_lower_char_vec();
        let g1 = EditTree::create_tree(&a, &b).unwrap();

        let f = "lachen".to_lower_char_vec();
        let f1 = "aaaaaaaaen".to_lower_char_vec();
        let exp = "gelacht".to_lower_char_vec();
        let exp1 = "geaaaaaaaat".to_lower_char_vec();

        assert_eq!(g.apply(&f).unwrap(), exp);
        assert_eq!(g1.apply(&f1).unwrap(), exp1);
        assert_eq!(g, g1);
    }

    #[test]
    fn test_graph_equality_and_applicability() {
        let mut set: HashSet<EditTree> = HashSet::default();
        let a = "abc".to_lower_char_vec();
        let b = "ab".to_lower_char_vec();
        let g1 = EditTree::create_tree(&a, &b).unwrap();

        let a = "aaa".to_lower_char_vec();
        let b = "aa".to_lower_char_vec();
        let g2 = EditTree::create_tree(&a, &b).unwrap();

        let a = "cba".to_lower_char_vec();
        let b = "ba".to_lower_char_vec();
        let g3 = EditTree::create_tree(&a, &b).unwrap();
        let g4 = EditTree::create_tree(&a, &b).unwrap();

        let a = "aaa".to_lower_char_vec();
        let b = "aac".to_lower_char_vec();
        let g5 = EditTree::create_tree(&a, &b).unwrap();

        let a = "dec".to_lower_char_vec();
        let b = "decc".to_lower_char_vec();
        let g6 = EditTree::create_tree(&a, &a).unwrap();
        let g7 = EditTree::create_tree(&a, &b).unwrap();

        set.insert(g1);
        assert_eq!(set.len(), 1);
        set.insert(g2);
        assert_eq!(set.len(), 2);
        set.insert(g3);
        assert_eq!(set.len(), 3);
        set.insert(g4);
        assert_eq!(set.len(), 3);
        set.insert(g5);
        assert_eq!(set.len(), 4);
        set.insert(g6);
        set.insert(g7);
        assert_eq!(set.len(), 6);

        let v = "yyyy".to_lower_char_vec();
        let res: HashSet<String> = set
            .iter()
            .map(|x| x.apply(&v))
            .filter(|x| x.is_some())
            .map(|x| x.unwrap().iter().collect::<String>())
            .collect();

        assert_eq!(res.len(), 2);

        let v = "yyy".to_lower_char_vec();
        let res: HashSet<String> = set
            .iter()
            .map(|x| x.apply(&v))
            .filter(|x| x.is_some())
            .map(|x| x.unwrap().iter().collect::<String>())
            .collect();
        assert!(res.contains("yyyc"));
        assert!(res.contains("yyy"));
        assert_eq!(res.len(), 2);

        let v = "bba".to_lower_char_vec();
        let res: HashSet<String> = set
            .iter()
            .map(|x| x.apply(&v))
            .filter(|x| x.is_some())
            .map(|x| x.unwrap().iter().collect::<String>())
            .collect();

        assert!(res.contains("bbac"));
        assert!(res.contains("bba"));
        assert!(res.contains("bb"));
        assert!(res.contains("bbc"));
        assert_eq!(res.len(), 4);

        let res: HashSet<String> = set
            .iter()
            .map(|x| x.apply(&a))
            .filter(|x| x.is_some())
            .map(|x| x.unwrap().iter().collect::<String>())
            .collect();
        assert!(res.contains("dec"));
        assert!(res.contains("decc"));
        assert!(res.contains("de"));
        assert_eq!(res.len(), 3);

        let a = "die".to_lower_char_vec();
        let b = "das".to_lower_char_vec();
        let c = "die".to_lower_char_vec();
        let g = EditTree::create_tree(&a, &b).unwrap();
        assert!(g.apply(&c).is_some());
    }
    #[test]
    fn test_graphs_inapplicable() {
        let g = EditTree::create_tree(&"abcdefg".to_lower_char_vec(), &"abc".to_lower_char_vec())
            .unwrap();
        assert!(g.apply(&"abc".to_lower_char_vec()).is_none());

        let g = EditTree::create_tree(&"abcdefg".to_lower_char_vec(), &"efg".to_lower_char_vec())
            .unwrap();
        assert!(g.apply(&"efg".to_lower_char_vec()).is_none());
    }
}
