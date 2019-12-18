extern crate pyo3;
extern crate rand;
use pyo3::prelude::*;
use pyo3::exceptions;
use pyo3::types::{PyDict, PyTuple};
use std::cmp;
use rand::Rng;

type Matrix = Vec<Vec<Vec<u8>>>;
type BatchMatrix = Vec<Vec<Vec<Vec<u8>>>>;

#[pyclass(module= "datastorage_lib")]
pub struct DataStorage {
    #[pyo3(get)]
    pub size: u32,
    #[pyo3(get)]
    pub is_visual: bool,
    #[pyo3(get)]
    pub frame_height: usize,
    #[pyo3(get)]
    pub frame_width: usize,
    #[pyo3(get)]
    pub agent_history_length: usize,
    #[pyo3(get)]
    pub batch_size: usize,
    #[pyo3(get)]
    pub count: usize,
    #[pyo3(get)]
    pub current: usize,
    #[pyo3(get)]
    pub actions: Vec<i32>,
    #[pyo3(get)]
    pub rewards: Vec<i32>,
    #[pyo3(get)]
    pub frames: Matrix,
    #[pyo3(get)]
    pub terminal_flags: Vec<bool>,
    #[pyo3(get)]
    pub indices: Vec<usize>,
    #[pyo3(get)]
    pub states: BatchMatrix,
    #[pyo3(get)]
    pub new_states: BatchMatrix,
}

#[pymethods]
impl DataStorage {
    #[new]
    #[args(size=100000, is_visual=true, frame_height=84, frame_width=84, agent_history_length=4, batch_size=4, kwargs="**")]
    pub fn new(obj: &PyRawObject, size: usize, is_visual: bool, frame_height: usize, frame_width: usize, agent_history_length: usize, batch_size: usize, kwargs: Option<&PyDict>){
        let mut actions = vec![0; size];
        let mut rewards = vec![0; size];
        let mut frames = vec![vec![vec![0; frame_width as usize]; frame_height as usize]; size];
        let mut terminal_flags = vec![false; size];
        let mut indices = vec![0; batch_size];
        let mut states = vec![vec![vec![vec![0; frame_width as usize]; frame_height as usize]; agent_history_length]; batch_size];
        let mut new_states = vec![vec![vec![vec![0; frame_width as usize]; frame_height as usize]; agent_history_length]; batch_size];
        obj.init(DataStorage{
            size: size as u32,
            is_visual: is_visual,
            frame_height: frame_height,
            frame_width: frame_width,
            agent_history_length: agent_history_length,
            batch_size: batch_size,
            count: 0,
            current: 0,
            actions: actions,
            rewards: rewards,
            frames: frames,
            terminal_flags: terminal_flags,
            indices: indices,
            states: states,
            new_states: new_states,
        });
    }

    fn add_experience(& mut self, action: i32, frame: Vec<Vec<u8>>, reward: i32, terminal: bool) -> PyResult<()> {
        let msg = format!("expected frames shape {:?}, found {:?}", (self.frames[0].len(), self.frames[0][0].len()),
                          (frame.len(), frame[0].len()));
        if frame.len() != self.frame_height {
            Err(exceptions::ValueError::py_err(msg))
        } else if frame[0].len() != self.frame_width  {
            Err(exceptions::ValueError::py_err(msg))
        } else {
            self.actions[self.current] = action;
            self.frames[self.current] = frame;
            self.rewards[self.current] = reward;
            self.terminal_flags[self.current] = terminal;
            self.count = cmp::max(self.count, self.current + 1);
            self.current = (self.current + 1) % self.size as usize;
            Ok(())
        }
    }

    fn _get_state(& self, index: usize) -> PyResult<Matrix> {
        if self.count == 0 {
            Err(exceptions::ValueError::py_err("The storage data is empty"))
        } else if index < (self.agent_history_length - 1) {
            Err(exceptions::ValueError::py_err(format!("Index must be min {}", self.agent_history_length -1)))
        } else {
            let frame = & self.frames[index + 1 - self.agent_history_length..index+1];
            Ok(frame.to_vec())
        }
    }

    fn _get_valid_indices(& mut self) {
        for i in 0..self.batch_size {
            let mut index: usize = 0;
            while true {
                index = rand::thread_rng().gen_range(self.agent_history_length, self.count -1);
                if index < self.agent_history_length {
                    continue;
                }
                if (index >= self.current) && ((index - self.agent_history_length) <= self.current) {
                    continue;
                }
                if self.terminal_flags[index - self.agent_history_length..index].contains(&true) {
                    continue;
                }
                break;
            }
            self.indices[i] = index;
        }
    }

    fn get_minibatch(& mut self) -> PyResult<(BatchMatrix, Vec<i32>, Vec<i32>, BatchMatrix, Vec<bool>)> {
        let mut actions = vec![];
        let mut rewards = vec![];
        let mut terminal_flags = vec![];

        if self.count < self.agent_history_length {
            Err(exceptions::ValueError::py_err("Not enough data in the storage to get a minibatch"))
        } else {
            self._get_valid_indices();
            for (i, idx) in self.indices.iter().enumerate() {

                match self._get_state(*idx-1) {
                    Err(err) => {
                        println!("Error retrieving data in the storage to get a minibatch");
                    },
                    Ok(t) => {
                        self.states[i] = t;
                    }
                }
                match self._get_state(*idx) {
                    Err(err) => {
                        println!("Error retrieving data in the storage to get a minibatch");
                    },
                    Ok(t) => {
                        self.new_states[i] = t;
                    }
                }
                // println!("Step - {}, action - {}", i, self.actions[*idx]);
                actions.push(self.actions[*idx]);
                rewards.push(self.rewards[*idx]);
                terminal_flags.push(self.terminal_flags[*idx]);
            }
            Ok(((self.states.to_vec(), actions, rewards, self.new_states.to_vec(), terminal_flags)))
        }
    }
}

#[pymodule]
fn datastoragelib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DataStorage>()?;
    Ok(())
}