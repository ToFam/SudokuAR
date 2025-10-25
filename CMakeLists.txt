cmake_minimum_required (VERSION 3.10 FATAL_ERROR)
project (sudoku-ar)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})

set(CMAKE_CXX_STANDARD 20)

if (WIN32)
else (WIN32)
    set (CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wall")
endif (WIN32)

find_package(OpenCL REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs video videoio)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CommonTarget sudokuar-common)
add_subdirectory (common)

OPTION(BUILD_TEST "Build test project for evaluation" ON)
OPTION(BUILD_TOOLS "Build development tools for algorithm parameter setup" OFF)

SET(AlgoTarget sudokuar-algo)
add_subdirectory(algo)

IF (BUILD_TEST)
    add_subdirectory(test)
ENDIF(BUILD_TEST)

IF (BUILD_TOOLS)
    add_subdirectory(tools)
ENDIF(BUILD_TOOLS)
