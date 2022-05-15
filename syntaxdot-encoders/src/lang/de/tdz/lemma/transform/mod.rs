use udgraph::graph::Sentence;

#[allow(clippy::len_without_is_empty)]
pub trait DependencyGraph {
    fn dependents<'a>(&'a self, idx: usize) -> Box<dyn Iterator<Item = (usize, String)> + 'a>;

    fn token(&self, idx: usize) -> &dyn Token;

    fn token_mut(&mut self, idx: usize) -> &mut dyn TokenMut;

    fn len(&self) -> usize;
}

impl DependencyGraph for Sentence {
    fn dependents<'a>(&'a self, idx: usize) -> Box<dyn Iterator<Item = (usize, String)> + 'a> {
        Box::new(self.dep_graph().dependents(idx).map(|triple| {
            (
                triple.dependent(),
                triple
                    .relation()
                    .expect("Edge without a dependency relation")
                    .to_owned(),
            )
        }))
    }

    fn token(&self, idx: usize) -> &dyn Token {
        self[idx]
            .token()
            .expect("The root node was used as a token")
    }

    fn token_mut(&mut self, idx: usize) -> &mut dyn TokenMut {
        self[idx]
            .token_mut()
            .expect("The root node was used as a token")
    }

    fn len(&self) -> usize {
        self.len()
    }
}

pub trait TokenMut: Token {
    fn set_lemma(&mut self, lemma: Option<String>);
}

pub trait Token {
    fn form(&self) -> &str;
    fn lemma(&self) -> &str;
    fn upos(&self) -> &str;
    fn xpos(&self) -> &str;
}

impl Token for udgraph::token::Token {
    fn form(&self) -> &str {
        self.form()
    }

    fn lemma(&self) -> &str {
        self.lemma().unwrap_or("_")
    }

    fn upos(&self) -> &str {
        self.upos().unwrap()
    }

    fn xpos(&self) -> &str {
        self.xpos().unwrap()
    }
}

impl TokenMut for udgraph::token::Token {
    fn set_lemma(&mut self, lemma: Option<String>) {
        self.set_lemma(lemma);
    }
}

pub trait Transform: Sync {
    fn transform(&self, graph: &dyn DependencyGraph, node: usize) -> String;
}

/// A list of `Transform`s.
pub struct Transforms(pub Vec<Box<dyn Transform>>);

impl Transforms {
    /// Transform a graph using the transformation list.
    ///
    /// This method applies the transformations to the given graph. Each
    /// transform is fully applied to the graph before the next transform,
    /// to ensure that dependencies between transforms are correctly handled.
    pub fn transform(&self, graph: &mut dyn DependencyGraph) {
        for t in &self.0 {
            for idx in 1..graph.len() {
                let lemma = t.as_ref().transform(graph, idx);
                graph.token_mut(idx).set_lemma(Some(lemma));
            }
        }
    }
}

pub mod delemmatization;

pub mod lemmatization;

pub mod misc;

mod named_entity;

mod svp;

#[cfg(test)]
pub(crate) mod test_helpers;
