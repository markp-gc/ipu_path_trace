cmake_minimum_required(VERSION 3.10)

project(keras_utils)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
add_compile_options(-Wall -Wextra -Werror -Wno-unused-parameter -Wno-ignored-qualifiers -Wno-sign-compare -Wno-error=deprecated-declarations)
set(CMAKE_CXX_FLAGS_DEBUG "-g -Og")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include_directories(${CMAKE_SOURCE_DIR})

file(GLOB K_HEADERS ${PROJECT_SOURCE_DIR}/*.hpp)
file(GLOB K_SRC ${PROJECT_SOURCE_DIR}/*.cpp)
add_library(keras_utils ${K_HEADERS} ${K_SRC})
