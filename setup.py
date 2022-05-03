from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension("fights_srv", "fights-srv/Cargo.toml", binding=Binding.PyO3)
    ],
)
