from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

setup_args = dict(
    ext_modules = [
        Pybind11Extension(
            # will be imported with this name
            "traceHMM.update",
            # locations of cpp files
            ["traceHMM/cpp/update.cpp"],
        ),
    ],
    # include to ensure updated C++ version
    cmdclass = {"build_ext": build_ext}
)

# append the arguments to the ones in pyproject.toml
setup(**setup_args)