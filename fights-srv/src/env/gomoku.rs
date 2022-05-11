use std::collections::HashMap;

use crate::{
    agent::Agent,
    env::{Env, Result},
    ndarray::NDArray,
};

pub struct GomokuEnv<'a> {
    pub win_condition: usize,
    pub board: NDArray<usize, 2>,
    pub size: (usize, usize),
    agents: HashMap<&'a Agent, usize>,
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
            let num = self.agents.get(agent).unwrap();
            self.board[[action[[0]], action[[1]]]] = num + 1;
        }

        Result {
            state: self.board.clone(),
            reward: 0.,
            done: self.check_win().is_some(),
            info: "Gomoku v0".to_string(),
        }
    }
}

impl<'a> GomokuEnv<'a> {
    pub fn new(
        (w, h): (usize, usize),
        win_condition: usize,
        agents: (&'a Agent, &'a Agent),
    ) -> GomokuEnv {
        let mut agents_hashmap = HashMap::new();

        agents_hashmap.insert(agents.0, 1);
        agents_hashmap.insert(agents.1, 2);

        GomokuEnv {
            win_condition,
            board: NDArray::from_iter(vec![0; w * h], &[w, h]).unwrap(),
            size: (w, h),
            agents: agents_hashmap,
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

    fn check_win(&self) -> Option<&Agent> {
        let kernels: Vec<NDArray<usize, 2>> = vec![
            NDArray::ones(&[1, self.win_condition]),
            NDArray::ones(&[self.win_condition, 1]),
            NDArray::identity(self.win_condition),
            NDArray::identity(self.win_condition).fliplr(),
        ];

        for (a, i) in &self.agents {
            for kernel in &kernels {
                let agent_board_vec: Vec<usize> = self
                    .board
                    .into_flat_vec()
                    .iter()
                    .map(|x| if x == i { 1 } else { 0 })
                    .collect();
                let agent_board = NDArray::from_vec(agent_board_vec, self.board.shape()).unwrap();
                let conv = agent_board.conv2d(&kernel).into_flat_vec();
                if conv.iter().any(|&x| x >= self.win_condition) {
                    return Some(a);
                }
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
    fn invalid_placement() {
        let a = Agent {
            id: "0".to_string(),
        };
        let b = Agent {
            id: "1".to_string(),
        };

        let env = GomokuEnv::new((10, 10), 5, (&a, &b));
        let valid = env.check_action(&NDArray::from_vec(vec![11, 0], &[2]).unwrap());
        assert_eq!(valid, false);
    }

    #[test]
    fn no_win() {
        let a = Agent {
            id: "0".to_string(),
        };
        let b = Agent {
            id: "1".to_string(),
        };

        let xs = [0, 2, 4, 5, 6];
        let mut env = GomokuEnv::new((10, 10), 5, (&a, &b));
        let mut res = env.reset();
        for x in xs {
            res = env.step(&a, NDArray::from_vec(vec![x, 0], &[2]).unwrap());
        }

        assert_eq!(res.done, false);
    }

    #[test]
    fn win_horizontal() {
        let a = Agent {
            id: "0".to_string(),
        };
        let b = Agent {
            id: "1".to_string(),
        };

        let xs = [0, 1, 2, 3, 4];
        let mut env = GomokuEnv::new((10, 10), 5, (&a, &b));
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
        let mut env = GomokuEnv::new((10, 10), 5, (&a, &b));
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

        let mut env = GomokuEnv::new((10, 10), 5, (&a, &b));
        env.step(&a, NDArray::from_vec(vec![0, 5], &[2]).unwrap());
        env.step(&a, NDArray::from_vec(vec![1, 4], &[2]).unwrap());
        env.step(&a, NDArray::from_vec(vec![2, 3], &[2]).unwrap());
        env.step(&a, NDArray::from_vec(vec![3, 2], &[2]).unwrap());
        let res = env.step(&a, NDArray::from_vec(vec![4, 1], &[2]).unwrap());
        assert_eq!(res.done, true);
    }

    #[test]
    fn win_diagonal_opposite() {
        let a = Agent {
            id: "0".to_string(),
        };
        let b = Agent {
            id: "1".to_string(),
        };

        let mut env = GomokuEnv::new((10, 10), 5, (&a, &b));
        env.step(&a, NDArray::from_vec(vec![4, 0], &[2]).unwrap());
        env.step(&a, NDArray::from_vec(vec![3, 1], &[2]).unwrap());
        env.step(&a, NDArray::from_vec(vec![2, 2], &[2]).unwrap());
        env.step(&a, NDArray::from_vec(vec![1, 3], &[2]).unwrap());
        let res = env.step(&a, NDArray::from_vec(vec![0, 4], &[2]).unwrap());
        assert_eq!(res.done, true);
    }

    #[test]
    #[should_panic]
    fn place_after_win() {
        let a = Agent {
            id: "0".to_string(),
        };
        let b = Agent {
            id: "1".to_string(),
        };

        let xs = [0, 1, 2, 3, 4];
        let mut env = GomokuEnv::new((10, 10), 5, (&a, &b));
        let mut res = env.reset();
        for x in xs {
            res = env.step(&a, NDArray::from_vec(vec![x, 0], &[2]).unwrap());
        }

        assert_eq!(res.done, true);
        _ = env.step(&b, NDArray::from_vec(vec![1, 1], &[1]).unwrap());
    }
}
