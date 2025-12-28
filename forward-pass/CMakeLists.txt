cmake_minimum_required(VERSION 3.16)

project(forward-pass LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(forward-pass
    src/main.cpp
)

target_include_directories(forward-pass
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
