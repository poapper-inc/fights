use std::{
    ops::{Index, IndexMut},
    vec::Vec,
};

pub struct NDArray<T, const N: usize> {
    data: Vec<T>,
    shape: [usize; N],
}

impl<T, const N: usize> NDArray<T, N>
where
    T: Clone,
{
    pub fn shape(&self) -> &[usize; N] {
        &self.shape
    }

    pub fn from_vec(data: Vec<T>, shape: &[usize; N]) -> Result<Self, ()> {
        match shape.iter().product::<usize>() {
            len if len == data.len() => Ok(NDArray {
                data: data,
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

impl<T, const N: usize> Index<&[usize; N]> for NDArray<T, N> {
    type Output = T;
    fn index<'a>(&'a self, idx: &[usize; N]) -> &'a T {
        let coeffs = self
            .shape
            .iter()
            .rev()
            .fold(vec![1], |mut acc, &x| -> Vec<usize> {
                acc.push(x * acc.last().unwrap().clone());
                acc
            });

        println!("{:?}", coeffs);

        let idx_flat: usize = idx.iter().rev().zip(&coeffs).map(|(x, y)| x * y).sum();
        &self.data[idx_flat]
    }
}

impl<T, const N: usize> IndexMut<&[usize; N]> for NDArray<T, N> {
    fn index_mut<'a>(&'a mut self, idx: &[usize; N]) -> &'a mut T {
        let coeffs = self
            .shape
            .iter()
            .rev()
            .fold(vec![1], |mut acc, &x| -> Vec<usize> {
                acc.push(x * acc.last().unwrap().clone());
                acc
            });

        let idx_flat: usize = idx.iter().rev().zip(&coeffs).map(|(x, y)| x * y).sum();
        self.data.index_mut(idx_flat)
    }
}

#[cfg(test)]
mod tests {
    use crate::ndarray::NDArray;

    #[test]
    fn new() {
        let iter = 0..1024;
        let arr_from_iter = NDArray::from_iter(iter, &[2, 8, 16, 4]).unwrap();
        assert_eq!(arr_from_iter.shape(), &[2, 8, 16, 4]);

        let arr_from_vec = NDArray::from_vec((0..1_00_00_00).collect(), &[100, 100, 100]).unwrap();
        assert_eq!(arr_from_vec.shape(), &[100, 100, 100]);

        let _arr_empty = NDArray::from_iter(&Vec::<f64>::new(), &[0]).unwrap();
    }

    #[test]
    #[should_panic]
    fn new_err() {
        let _arr = NDArray::from_iter(0..12, &[3, 2, 5]).unwrap();
    }

    #[test]
    fn index() {
        let arr = NDArray::from_iter(0..24, &[3, 2, 2, 2]).unwrap();
        assert_eq!(arr[&[0, 0, 0, 0]], 0);
        assert_eq!(arr[&[2, 1, 0, 1]], 21);
        assert_eq!(arr[&[2, 1, 1, 1]], 23);
    }

    #[test]
    fn index_mut() {
        let mut arr = NDArray::from_iter(0..24, &[3, 2, 2, 2]).unwrap();
        let x = &mut arr[&[2, 1, 0, 1]];
        *x = 0;

        assert_eq!(arr[&[2, 1, 0, 1]], 0);
    }
}
