# fights

[![Test CI](https://github.com/poapper-inc/fights/actions/workflows/test.yml/badge.svg)](https://github.com/poapper-inc/fights/actions/workflows/test.yml)
[![Lint CI](https://github.com/poapper-inc/fights/actions/workflows/lint.yml/badge.svg)](https://github.com/poapper-inc/fights/actions/workflows/lint.yml)

_Work in progress_

Competitive artificial battle environments.

Consists of:

- [fights](/fights): Python package for environments
- [fights-srv](/fights-srv): Rust game server for state calculation
- [fights-rust](/fights-rust): Rust extension for Python

## Setup

Requires Python and Rust toolchains to be installed.

```shell
$ python3 -m pip install tox
$ tox -e dev --devenv env
$ source env/bin/activate
```

## Usage

### fights + fights-rust

```shell
# build wheel
$ python -m build
```

### fights-srv

```shell
# development build
$ cargo build

# production build
$ cargo build --release

# tests
$ cargo test

# run
$ cargo run
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
