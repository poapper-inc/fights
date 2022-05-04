use crate::ndarray::NDArray;

type Action<const A: usize> = NDArray<f64, A>;

pub trait Env<const A: usize, const S: usize, const R: usize> {
    fn step(action: Action<A>) -> Result<S, R>;
}

pub struct Result<const S: usize, const R: usize> {
    pub state: NDArray<f64, S>,
    pub reward: NDArray<f64, R>,
    pub done: bool,
    pub info: String,
}
