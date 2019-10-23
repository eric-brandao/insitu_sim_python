cmake_minimum_required(VERSION 2.8.12)
set(CMAKE_BUILD_TYPE Release)
project(insitu_cpp)

file(GLOB_RECURSE INSITU_CPP_SOURCES "insitu_cpp/src/*.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/insitu_cpp/include)

# set(PYBIND11_CPP_STANDARD -std=c++14)
set(PYBIND11_CPP_STANDARD -std=c++1y)  # to be able to build on CentOS_5
add_subdirectory(pybind11)
pybind11_add_module(insitu_cpp "${INSITU_CPP_SOURCES}")

