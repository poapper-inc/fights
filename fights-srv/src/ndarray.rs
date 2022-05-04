use std::{
    ops::{Index, IndexMut},
    vec::Vec,
};

pub struct NDArray<T, const N: usize> {
    pub data: Vec<T>,
    pub shape: [usize; N],
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
    fn index() {
        let arr = NDArray {
            data: (0..24).collect(),
            shape: [3, 2, 2, 2],
        };
        assert_eq!(arr[&[0, 0, 0, 0]], 0);
        assert_eq!(arr[&[2, 1, 0, 1]], 21);
        assert_eq!(arr[&[2, 1, 1, 1]], 23);
    }

    #[test]
    fn index_mut() {
        let mut arr = NDArray {
            data: (0..24).collect(),
            shape: [3, 2, 2, 2],
        };
        let x = &mut arr[&[2, 1, 0, 1]];
        *x = 0;

        assert_eq!(arr[&[2, 1, 0, 1]], 0);
    }
}
