Installation
============

For current version, please clone the github repository to your machine and enter the directory by
```bash
git clone https://github.com/hyuyu104/TraceModel .
cd TraceModel
```
Then either writing codes directly in the `TraceModel` directory or installing the package locally so that program using the environment has access to the `traceHMM` package. This can be accomplished either by `conda`:
```bash
conda create --name trace_env python==3.12.2
conda activate trace_env
python -m pip install -e .
```
or `venv`:
```bash
python -m venv trace_venv
source trace_venv/bin/activate
python -m pip install -e .
```
However, in both cases, please ensure the Python version is  at least`3.10`.