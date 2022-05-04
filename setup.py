from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension("fights_rust", "fights-rust/Cargo.toml", binding=Binding.PyO3)
    ],
)
