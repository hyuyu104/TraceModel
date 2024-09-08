cmake_minimum_required(VERSION 3.12)
project(update)

set(PYTHON_EXECUTABLE /opt/miniconda3/bin/python)

set(CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED)

pybind11_add_module(update update.cpp)