cmake_minimum_required(VERSION 3.15)
project(LLMEngine)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

execute_process(COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
                OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(pybind11 REQUIRED)

pybind11_add_module(tiny_math LLMEngine.cpp)

target_include_directories(tiny_math PRIVATE ${CMAKE_SOURCE_DIR}/lib/eigen)

set(OUTPUT_DIR "${CMAKE_SOURCE_DIR}/..")

set_target_properties(tiny_math PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE "${OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG "${OUTPUT_DIR}"
)