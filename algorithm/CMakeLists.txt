cmake_minimum_required(VERSION 3.27)
project(NM_lab_Simp_Iter_Method)

set(CMAKE_CXX_STANDARD 17)

add_library(libNM1_lib SHARED library.cpp )
set(LIBRARY_OUTPUT_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../dll)