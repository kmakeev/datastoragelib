extern crate pyo3;
extern crate rand;
extern crate rayon;
use pyo3::prelude::*;
use pyo3::exceptions;
use pyo3::types::{PyDict};
use std::cmp;
use rand::Rng;
use rayon::prelude::*;
use ndarray::{Array1, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray4};

// type Matrix = Vec<Vec<Vec<u8>>>;
// type BatchMatrix = Vec<Vec<Vec<Vec<u8>>>>;

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
    pub indices: Vec<usize>,
    pub actions: Vec<i32>,
    pub rewards: Vec<i32>,
    // pub frames: Matrix,
    pub frames: Vec<Vec<u8>>,
    pub terminal_flags: Vec<bool>,
    // pub states: Vec<u8>,
    // pub new_states: Vec<u8>,
    #[pyo3(get)]
    pub shape: (usize, usize, usize, usize),
    //pub actions: Array1<i32>,
    //pub rewards: Array1<i32>,
    //pub frames: Array3<u8>,
    //pub terminal_flags: Array1<bool>,
    //pub states: Array4<u8>,
    //pub states: BatchMatrix,
    //pub new_states: Array4<u8>,
}

#[pymethods]
impl DataStorage {
    #[new]
    #[args(size = 100000, is_visual = true, frame_height = 84, frame_width = 84, agent_history_length = 4, batch_size = 4, kwargs = "**")]
    pub fn new(obj: &PyRawObject, size: usize, is_visual: bool, frame_height: usize, frame_width: usize, agent_history_length: usize, batch_size: usize, _kwargs: Option<&PyDict>) {
        // let actions = Array1::zeros(size);
        //let rewards = Array1::zeros(size);
        //let frames = Array3::zeros((frame_width, frame_height, size));
        //let terminal_flags = Array1::from_elem(size, false);
        //let states = Array4::zeros((frame_width, frame_height, agent_history_length, batch_size));
        //let states = vec![vec![vec![vec![0; frame_width as usize]; frame_height as usize]; agent_history_length]; batch_size];

        //let new_states = Array4::zeros((frame_width, frame_height, agent_history_length, batch_size));
        let actions = vec![0; size];
        let rewards = vec![0; size];
        // let frames = vec![vec![vec![0; frame_width as usize]; frame_height as usize]; size];
        let frames = vec![vec![0; frame_width * frame_height as usize]; size];
        let terminal_flags = vec![false; size];
        let indices = vec![0; batch_size];
        let shape = (batch_size, agent_history_length, frame_height, frame_width);
        // let states = vec![0; batch_size*frame_height*frame_height];
        // let new_states = vec![0; batch_size*frame_height*frame_height];
        // let states = vec![vec![vec![vec![0; frame_width as usize]; frame_height as usize]; agent_history_length]; batch_size];
        // let new_states = vec![vec![vec![vec![0; frame_width as usize]; frame_height as usize]; agent_history_length]; batch_size];

        obj.init(DataStorage {
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
            shape: shape,
            // states: states,
            //new_states: new_states,
        });
    }

    fn add_experience(& mut self, action: i32, frame: Vec<Vec<u8>>, reward: i32, terminal: bool) -> PyResult<()> {
        let msg = format!("expected frames size != height*width {}, found {}", self.frames.len(), frame.len());
        if frame.len() != self.frame_height {
            Err(exceptions::ValueError::py_err(msg))
        } else if frame[0].len() != self.frame_width  {
            Err(exceptions::ValueError::py_err(msg))
        } else {
            self.actions[self.current] = action;
            self.frames[self.current] = frame.par_iter().flat_map(|_i| _i.clone()).collect();
            self.rewards[self.current] = reward;
            self.terminal_flags[self.current] = terminal;
            self.count = cmp::max(self.count, self.current + 1);
            self.current = (self.current + 1) % self.size as usize;
            Ok(())
        }
    }

    fn get_minibatch(& mut self, _py: Python) -> PyResult<(Py<PyArray4<u8>>, Py<PyArray1<i32>>, Py<PyArray1<i32>>, Py<PyArray4<u8>>, Py<PyArray1<bool>>)> {
        let mut actions = vec![];
        let mut rewards = vec![];
        let mut terminal_flags = vec![];
        let mut states = vec![];
        let mut new_states = vec![];

        // let mut array_test = vec!(0; 4*84*84);

        if self.count < self.agent_history_length {
            Err(exceptions::ValueError::py_err("Not enough data in the storage to get a minibatch"))
        } else {
            self._get_valid_indices();

            for idx in self.indices.iter() {

                match self._get_state(*idx-1) {
                    Err(_err) => {
                        println!("Error retrieving data in the storage to get a minibatch");
                    },
                    Ok(t) => {

                        states.extend_from_slice(&t);
                        //states.extend_from_slice(&array_test);
                    }
                }
                match self._get_state(*idx) {
                    Err(_err) => {
                        println!("Error retrieving data in the storage to get a minibatch");
                    },
                    Ok(t) => {
                        // new_states.extend(t);
                        new_states.extend_from_slice(&t);
                    }
                }
                // println!("Step - {}, action - {}", i, self.actions[*idx]);
                actions.push(self.actions[*idx]);
                rewards.push(self.rewards[*idx]);
                terminal_flags.push(self.terminal_flags[*idx]);
            }
            let st = Array4::from_shape_vec(self.shape, states).unwrap();
            let new_st = Array4::from_shape_vec(self.shape, new_states).unwrap();

            Ok((st.permuted_axes([0, 2, 3, 1]).into_pyarray(_py).to_owned(), Array1::from(actions).into_pyarray(_py).to_owned(),  Array1::from(rewards).into_pyarray(_py).to_owned(),
                new_st.permuted_axes([0, 2, 3, 1]).into_pyarray(_py).to_owned(), Array1::from(terminal_flags).into_pyarray(_py).to_owned()))
        }
    }
}


impl DataStorage {

    /*
    fn _get_state(& self, index: usize) -> Result<Array3<u8>, String> {
        if self.count == 0 {
            Err(String::from("The storage data is empty"))
        } else if index < (self.agent_history_length - 1) {
            Err(format!("Index must be min {}", self.agent_history_length -1))
        } else {
            let frame = & self.frames.slice(s![index + 1 - self.agent_history_length..index+1, .. , .. ]);
            Ok(frame.into_owned())
        }
    }
    */

    fn _get_state(& self, index: usize) -> Result<Vec<u8>, String> {
        if self.count == 0 {
            Err(String::from("The storage data is empty"))
        } else if index < (self.agent_history_length - 1) {
            Err(format!("Index must be min {}", self.agent_history_length -1))
        } else {
            let frame = & self.frames[index + 1 - self.agent_history_length..index+1];
            Ok(frame.concat())
        }
    }

    fn _get_valid_indices(& mut self) {
        // let range = std::ops::Range {start: 0, end: self.batch_size as u32};
        let range = vec![0; self.batch_size];
        let mut result: Vec<usize> = vec![];
        range.par_iter().map (|_i| {
            let mut index;
            loop {
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
            index
        }).collect_into_vec(& mut result);
        self.indices = result;
    }
}



#[pymodule]
fn datastoragelib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DataStorage>()?;
    Ok(())
}