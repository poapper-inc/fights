use crate::{
    agent::Agent,
    env::{Env, Result},
    ndarray::NDArray,
};

pub struct GomokuEnv<'a> {
    pub win_condition: usize,
    pub board: NDArray<usize, 2>,
    pub size: (usize, usize),
    pub agents: [&'a Agent; 2],
}

impl Env for GomokuEnv<'_> {
    type Action = NDArray<usize, 1>;
    type State = NDArray<usize, 2>;

    fn reset(&mut self) -> Result<Self::State> {
        self.board = NDArray::<usize, 2>::from_vec(
            vec![0; self.size.0 * self.size.1],
            &[self.size.0, self.size.1],
        )
        .unwrap();

        Result {
            state: self.board.clone(),
            reward: 0.,
            done: false,
            info: "Gomoku v0".to_string(),
        }
    }

    fn step(&mut self, agent: &Agent, action: Self::Action) -> Result<Self::State> {
        if self.check_action(&action) {
            let num = self.agents.iter().position(|&a| a == agent).unwrap();
            self.board[[action[[0]], action[[1]]]] = num + 1;
        }

        Result {
            state: self.board.clone(),
            reward: 0.,
            done: self.check_win(agent).is_some(),
            info: "Gomoku v0".to_string(),
        }
    }
}

impl<'a> GomokuEnv<'a> {
    pub fn new((w, h): (usize, usize), win_condition: usize, agents: [&'a Agent; 2]) -> GomokuEnv {
        GomokuEnv {
            win_condition,
            board: NDArray::from_iter(vec![0; w * h], &[w, h]).unwrap(),
            size: (w, h),
            agents,
        }
    }

    fn check_action(&self, action: &<Self as Env>::Action) -> bool {
        let pos = [action[[0]], action[[1]]];
        match pos {
            [x, _y] if !(0..self.size.0).contains(&x) => false,
            [_x, y] if !(0..self.size.1).contains(&y) => false,
            [x, y] if self.board[[x, y]] != 0 => false,
            _ => true,
        }
    }

    fn check_win(&self, agent: &'a Agent) -> Option<&Agent> {
        let kernels: Vec<NDArray<usize, 2>> = vec![
            NDArray::ones(&[1, self.win_condition]),
            NDArray::ones(&[self.win_condition, 1]),
            NDArray::eye(self.win_condition),
            NDArray::eye(self.win_condition).fliplr(),
        ];

        for kernel in kernels {
            let conv = self.board.conv2d(kernel).into_flat_vec();

            if conv.iter().any(|&x| x >= 5) {
                return Some(agent);
            }
        }

        return None;
    }
}

#[cfg(test)]
mod tests {
    use crate::{agent::Agent, env::Env, ndarray::NDArray};

    use super::GomokuEnv;

    #[test]
    fn win_horizontal() {
        let a = Agent {
            id: "0".to_string(),
        };
        let b = Agent {
            id: "1".to_string(),
        };

        let xs = [0, 1, 2, 3, 4];
        let mut env = GomokuEnv::new((10, 10), 5, [&a, &b]);
        let mut res = env.reset();
        for x in xs {
            res = env.step(&a, NDArray::from_vec(vec![x, 0], &[2]).unwrap());
        }

        assert_eq!(res.done, true);
    }

    #[test]
    fn win_vertical() {
        let a = Agent {
            id: "0".to_string(),
        };
        let b = Agent {
            id: "1".to_string(),
        };

        let ys = [0, 1, 2, 3, 4];
        let mut env = GomokuEnv::new((10, 10), 5, [&a, &b]);
        let mut res = env.reset();
        for y in ys {
            res = env.step(&a, NDArray::from_vec(vec![0, y], &[2]).unwrap());
        }

        assert_eq!(res.done, true);
    }

    #[test]
    fn win_diagonal() {
        let a = Agent {
            id: "0".to_string(),
        };
        let b = Agent {
            id: "1".to_string(),
        };

        let mut env = GomokuEnv::new((10, 10), 5, [&a, &b]);
        env.step(&a, NDArray::from_vec(vec![0, 5], &[2]).unwrap());
        env.step(&a, NDArray::from_vec(vec![1, 4], &[2]).unwrap());
        env.step(&a, NDArray::from_vec(vec![2, 3], &[2]).unwrap());
        env.step(&a, NDArray::from_vec(vec![3, 2], &[2]).unwrap());
        let res = env.step(&a, NDArray::from_vec(vec![4, 1], &[2]).unwrap());
        assert_eq!(res.done, true);
    }
}
