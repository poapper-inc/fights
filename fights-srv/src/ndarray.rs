use num_traits::Num;
use std::{
    fmt::Debug,
    ops::{AddAssign, Index, IndexMut},
    vec::Vec,
};

#[derive(Clone, Debug)]
pub struct NDArray<T: Num, const N: usize> {
    data: Vec<T>,
    shape: [usize; N],
}

impl<T> NDArray<T, 2>
where
    T: Clone + Num + AddAssign,
{
    pub fn eye(cols: usize, rows: usize) -> Self {
        let mut result = NDArray::from_vec(vec![T::zero(); rows * cols], &[cols, rows]).unwrap();
        for i in 0..rows.min(cols) {
            result[[i, i]] = T::one();
        }
        result
    }

    pub fn identity(rows: usize) -> Self {
        NDArray::eye(rows, rows)
    }

    pub fn transpose(&self) -> Self {
        let mut result = self.clone();
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                result[[j, i]] = self[[i, j]].clone();
            }
        }
        result
    }

    pub fn fliplr(&self) -> Self {
        let mut result = self.clone();
        for i in 0..self.shape[0] {
            for j in 0..self.shape[1] {
                result[[i, self.shape[1] - j - 1]] = self[[i, j]].clone();
            }
        }
        result
    }

    pub fn conv2d(&self, kernel: &NDArray<T, 2>) -> Self {
        let mut result = NDArray::zeros(&[self.shape[0], self.shape[1]]);
        for i in 0..self.shape[0] - kernel.shape[0] + 1 {
            for j in 0..self.shape[1] - kernel.shape[1] + 1 {
                let mut acc = T::zero();
                for k in 0..kernel.shape[0] {
                    for l in 0..kernel.shape[1] {
                        acc += kernel[[k, l]].clone() * self[[i + k, j + l]].clone();
                    }
                }
                result[[i, j]] = acc;
            }
        }
        result
    }
}

impl<T, const N: usize> NDArray<T, N>
where
    T: Clone + Num,
{
    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    pub fn new() -> Self {
        NDArray::from_vec(vec![], &[0; N]).unwrap()
    }

    pub fn zeros(shape: &[usize; N]) -> Self {
        let n = shape.iter().product();
        NDArray::from_vec(vec![T::zero(); n], shape).unwrap()
    }

    pub fn ones(shape: &[usize; N]) -> Self {
        let n = shape.iter().product();
        NDArray::from_vec(vec![T::one(); n], shape).unwrap()
    }

    pub fn into_flat_vec(&self) -> Vec<T> {
        return self.data.clone();
    }

    pub fn from_vec(data: Vec<T>, shape: &[usize; N]) -> Result<Self, ()> {
        match shape.iter().product::<usize>() {
            len if len == data.len() => Ok(NDArray {
                data,
                shape: *shape,
            }),
            _ => Err(()),
        }
    }

    pub fn from_iter(data: impl IntoIterator<Item = T>, shape: &[usize; N]) -> Result<Self, ()> {
        let v: Vec<T> = data.into_iter().collect();
        match shape.iter().product::<usize>() {
            len if len == v.len() => Ok(NDArray {
                data: v,
                shape: *shape,
            }),
            _ => Err(()),
        }
    }
}

impl<T, const N: usize> NDArray<T, N>
where
    T: Num,
{
    fn flatten_index(&self, idx: [usize; N]) -> usize {
        let coeffs = self
            .shape
            .iter()
            .rev()
            .fold(vec![1], |mut acc, &x| -> Vec<usize> {
                acc.push(x * acc.last().unwrap().clone());
                acc
            });
        idx.iter().rev().zip(&coeffs).map(|(x, y)| x * y).sum()
    }
}

impl<T: Num, const N: usize> Index<[usize; N]> for NDArray<T, N> {
    type Output = T;
    fn index<'a>(&'a self, idx: [usize; N]) -> &'a T {
        let idx_flat: usize = self.flatten_index(idx);
        &self.data[idx_flat]
    }
}

impl<T: Num, const N: usize> IndexMut<[usize; N]> for NDArray<T, N> {
    fn index_mut<'a>(&'a mut self, idx: [usize; N]) -> &'a mut T {
        let idx_flat: usize = self.flatten_index(idx);
        self.data.index_mut(idx_flat)
    }
}

#[cfg(test)]
mod tests {
    use crate::ndarray::NDArray;

    #[test]
    fn new() {
        let iter = 0..1024i64;
        let arr_from_iter = NDArray::from_iter(iter, &[2, 8, 16, 4]).unwrap();
        assert_eq!(arr_from_iter.shape(), &[2, 8, 16, 4]);

        let arr_from_vec =
            NDArray::from_vec((0..1_00_00_00i64).collect(), &[100, 100, 100]).unwrap();
        assert_eq!(arr_from_vec.shape(), &[100, 100, 100]);

        let _arr_empty = NDArray::from_iter(Vec::<f64>::new(), &[0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn new_err() {
        let _arr = NDArray::from_iter(0..12, &[3, 2, 5]).unwrap();
    }

    #[test]
    fn eye() {
        let cols = 3usize;
        let rows = 4usize;
        let arr: NDArray<usize, 2> = NDArray::eye(cols, rows);
        for x in 0..cols {
            for y in 0..rows {
                assert_eq!(arr[[x, y]], if x == y { 1 } else { 0 });
            }
        }
    }

    #[test]
    fn identity() {
        let rows = 4usize;
        let arr: NDArray<usize, 2> = NDArray::identity(rows);
        for x in 0..rows {
            for y in 0..rows {
                assert_eq!(arr[[x, y]], if x == y { 1 } else { 0 });
            }
        }
    }

    #[test]
    fn index() {
        let arr = NDArray::from_iter(0..24, &[3, 2, 2, 2]).unwrap();
        assert_eq!(arr[[0, 0, 0, 0]], 0);
        assert_eq!(arr[[2, 1, 0, 1]], 21);
        assert_eq!(arr[[2, 1, 1, 1]], 23);
    }

    #[test]
    fn index_mut() {
        let mut arr = NDArray::from_iter(0..24, &[3, 2, 2, 2]).unwrap();
        let x = &mut arr[[2, 1, 0, 1]];
        *x = 0;

        assert_eq!(arr[[2, 1, 0, 1]], 0);
    }
}
