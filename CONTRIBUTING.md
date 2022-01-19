# Contributing

## Conventions

### Code

All code must follow [PEP 8](https://www.python.org/dev/peps/pep-0008/), except for the maximum line length, which is set to be 88, following `black`'s rules.

Developers can ensure adherence to PEP 8 by running [formatters](#formatting) and [linters](#linting), as described below.

### Language

All comments, documentation, commit messages, issues, etc. must be in English.

## Setup

Python >= 3.7 is required.

Dependencies needed in development are specified in [requirements-dev.txt](requirements-dev.txt).

A virual environment with all development dependencies can be created with `tox`:

```shell
$ python -m pip install tox
$ tox -e dev --devenv env
$ source env/bin/activate
```

Alternatively, a virtual environment can be created manually with `venv`:

```shell
$ python -m venv env
$ source env/bin/activate
$ python -m pip install -r requirements-dev.txt
```

## Testing

The full test suite can be run with `tox`:

```shell
$ tox
```

Alternatively, tests for a single version of Python can be run directly via `pytest`:

```shell
$ pytest
```

## Formatting

Format all code:
```shell
$ tox -e format
```

## Linting

Lint all code:
```shell
$ tox -e lint
```

## Type checking

Check types:
```shell
$ tox -e typecheck
```

## Building

This package follows [PEP 517](https://www.python.org/dev/peps/pep-0517/) and [PEP 518](https://www.python.org/dev/peps/pep-0518/).

```shell
$ python -m pip install build
$ python -m build
```
