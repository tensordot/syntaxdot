use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use petgraph::Direction;

use super::{DependencyGraph, Token, TokenMut, Transform};

pub struct TestCase {
    graph: TestCaseGraph,
    index: usize,
    correct: String,
}

struct TestCaseGraph(pub DiGraph<TestToken, String>);

impl DependencyGraph for TestCaseGraph {
    fn dependents<'a>(&'a self, idx: usize) -> Box<dyn Iterator<Item = (usize, String)> + 'a> {
        Box::new(
            self.0
                .edges_directed(NodeIndex::new(idx), Direction::Outgoing)
                .map(|e| (e.target().index(), e.weight().to_owned())),
        )
    }

    fn token(&self, idx: usize) -> &dyn Token {
        &self.0[NodeIndex::new(idx)]
    }

    fn token_mut(&mut self, idx: usize) -> &mut dyn TokenMut {
        &mut self.0[NodeIndex::new(idx)]
    }

    fn len(&self) -> usize {
        self.0.node_count()
    }
}

pub struct TestToken {
    form: String,
    lemma: String,
    upos: String,
    xpos: String,
}

impl Token for TestToken {
    fn form(&self) -> &str {
        &self.form
    }

    fn lemma(&self) -> &str {
        &self.lemma
    }

    fn upos(&self) -> &str {
        &self.upos
    }

    fn xpos(&self) -> &str {
        &self.xpos
    }
}

impl TokenMut for TestToken {
    fn set_lemma(&mut self, lemma: Option<String>) {
        self.lemma = lemma.expect("Missing lemma for test token");
    }
}

fn read_dependency(iter: &mut dyn Iterator<Item = &str>) -> Option<(String, TestToken)> {
    // If there is a relation, read it, otherwise bail out.
    let rel = iter.next()?.to_owned();

    // However, if there is a relation and no token, panic.
    Some((
        rel,
        read_token(iter).expect("Incomplete dependency relation"),
    ))
}

fn read_token(iter: &mut dyn Iterator<Item = &str>) -> Option<TestToken> {
    Some(TestToken {
        form: iter.next()?.to_owned(),
        lemma: iter.next()?.to_owned(),
        upos: iter.next()?.to_owned(),
        xpos: iter.next()?.to_owned(),
    })
}

fn read_test_cases<R>(buf_read: R) -> Vec<TestCase>
where
    R: BufRead,
{
    let mut test_cases = Vec::new();

    for line in buf_read.lines() {
        let line = line.unwrap();
        let line_str = line.trim();

        // Skip empty lines
        if line_str.is_empty() {
            continue;
        }

        // Skip comments
        if line_str.starts_with('#') {
            continue;
        }

        let mut iter = line.split_whitespace();

        let mut graph = DiGraph::new();

        graph.add_node(TestToken {
            form: "ROOT".to_string(),
            lemma: "ROOT".to_string(),
            upos: "root".to_string(),
            xpos: "root".to_string(),
        });

        let test_token = read_token(&mut iter).unwrap();
        let index = graph.add_node(test_token);
        let correct = iter
            .next()
            .unwrap_or_else(|| panic!("Gold standard lemma missing: {}", line_str))
            .to_owned();

        // Optional: read head
        if let Some((rel, head)) = read_dependency(&mut iter) {
            let head_index = graph.add_node(head);
            graph.add_edge(head_index, index, rel);
        }

        // Optional: read dependents
        while let Some((rel, dep)) = read_dependency(&mut iter) {
            let dep_index = graph.add_node(dep);
            graph.add_edge(index, dep_index, rel);
        }

        let test_case = TestCase {
            graph: TestCaseGraph(graph),
            index: index.index(),
            correct,
        };

        test_cases.push(test_case);
    }

    test_cases
}

pub fn run_test_cases<P, T>(filename: P, transform: T)
where
    P: AsRef<Path>,
    T: Transform,
{
    let f = File::open(filename).unwrap();
    let test_cases = read_test_cases(BufReader::new(f));

    for test_case in test_cases {
        assert_eq!(
            test_case.correct,
            transform.transform(&test_case.graph, test_case.index)
        )
    }
}
