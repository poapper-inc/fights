use pyo3::prelude::*;

#[pyo3::pyfunction]
fn return_one() -> i32 {
    1
}

#[pymodule]
fn fights_rust(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(return_one, m)?)?;
    Ok(())
}
