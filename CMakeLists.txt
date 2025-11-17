cmake_minimum_required(VERSION 3.16)

if (NOT DEFINED SKBUILD_PROJECT_NAME)
    set(SKBUILD_PROJECT_NAME "cpyutl")
endif ()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C)
set(CMAKE_C_STANDARD 17)

find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
execute_process(
        COMMAND "${Python_EXECUTABLE}"
        -c "import numpy; print(numpy.get_include())"
        OUTPUT_VARIABLE NumPy_INCLUDE_DIRS
        OUTPUT_STRIP_TRAILING_WHITESPACE
)

if (NumPy_INCLUDE_DIRS STREQUAL "")
    message(FATAL_ERROR "NumPy include directory not found.")
endif()


add_library(cpyutl STATIC src/cpyutl.c)
target_include_directories(cpyutl PRIVATE ${Python_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})

if (CMAKE_C_COMPILER_ID STREQUAL "GNU")
    target_compile_options(cpyutl PRIVATE -Wall -Wextra -Werror)
endif()

add_subdirectory(src/python)
