use pyo3::prelude::*;
use std::collections::HashMap;

struct TrieNode {
    children: HashMap<char, TrieNode>,
    token_id: Option<usize>,
}

impl TrieNode {
    fn new() -> Self {
        TrieNode {
            children: HashMap::new(),
            token_id: None,
        }
    }
}

#[pyclass]
pub struct Trie {
    root: TrieNode,
    next_id: usize, // for assigning unique IDs to tokens
}

#[pymethods]
impl Trie {
    #[new]
    pub fn new() -> Self {
        Trie {
            root: TrieNode::new(),
            next_id: 0, // We start the IDs at 0
        }
    }

    pub fn add(&mut self, word: &str) {
        let mut node = &mut self.root;
        for ch in word.chars() {
            node = node.children.entry(ch).or_insert_with(TrieNode::new);
        }
        if node.token_id.is_none() {
            node.token_id = Some(self.next_id);
            self.next_id += 1;
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<usize> {
        let mut tokens = vec![];
        let mut start = 0;
        
        while start < text.len() {
            let mut node = &self.root;
            let mut matched = false;
            let mut end = start;
            for ch in text[start..].chars() {
                if let Some(next_node) = node.children.get(&ch) {
                    node = next_node;
                    end += ch.len_utf8();
                    if node.token_id.is_some() {
                        matched = true;
                        break;
                    }
                } else {
                    break;
                }
            }
            
            if matched {
                tokens.push(node.token_id.unwrap());
                start = end;
            } else {
                tokens.push(self.next_id); // Assign unknown token ID
                start += text[start..].chars().next().unwrap().len_utf8();
            }
        }

        tokens
    }
}

#[pymodule]
fn rust_trie(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Trie>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;

    #[test]
    fn test_trie() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let trie_module = PyModule::new(py, "trie_module").unwrap();
        let locals = [("trie", trie_module)].into_py_dict(py);
        let py_trie: PyObject = py
            .eval("trie.Trie()", Some(locals), None)
            .unwrap()
            .into();
        py_trie.call_method0("add", "[CLS]").unwrap();
        let tokens: Vec<usize> = py_trie
            .call_method1("tokenize", ("[CLS] This is a test",))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(tokens, vec![0, 1, 1, 1, 1]);
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::types::IntoPyDict;

    #[test]
    fn test_trie() {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let trie_mod = PyModule::new(py, "trie_module").unwrap();
        let locals = [("trie", trie_mod)].into_py_dict(py);
        let py_trie: PyObject = py
            .eval("trie.Trie()", Some(locals), None)
            .unwrap()
            .into();
        py_trie.call_method0("add", "[CLS]").unwrap();
        let tokens: Vec<usize> = py_trie
            .call_method1("tokenize", ("[CLS] This is a test",))
            .unwrap()
            .extract()
            .unwrap();
        assert_eq!(tokens, vec![0, 1, 1, 1, 1]);
    }
}

