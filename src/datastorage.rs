extern crate pyo3;
use pyo3::prelude::*;
// use pyo3::wrap_function;
use pyo3::types::{PyDict, PyTuple};


#[pyclass(module= "datastorage_lib")]
pub struct DataStorage {
    pub size: usize,
    pub is_visual: bool,
    pub frame_height: u32,
    pub frame_width: u32,
    pub agent_history_length: u32,
    pub batch_size: u32,
    pub count: u32,
    pub current: u32,
    pub actions: Vec<i32>,
    pub rewards: Vec<i32>,
    pub frames: Vec<i8>,
    pub terminal_flags: Vec<bool>,
}

#[pymethods]
impl DataStorage {
    #[new]
    #[args(size=1000000, is_visual=true, frame_height=84, frame_width=84, agent_history_length=4, batch_size=4, kwargs="**")]
    pub fn new(obj: &PyRawObject, size: usize, is_visual: bool, frame_height: u32, frame_width: u32, agent_history_length: u32, batch_size: u32, kwargs: Option<&PyDict>){
        let mut actions = vec![0; size];
        let mut rewards = vec![0; size];
        let mut frames = vec![0; (size as u32 * frame_width * frame_height) as usize];

        let mut terminal_flags = vec![false; size];
        obj.init(DataStorage{size: size,
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
        });
    }
}

#[pymodule]
fn datastoragelib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<DataStorage>()?;
    Ok(())
}