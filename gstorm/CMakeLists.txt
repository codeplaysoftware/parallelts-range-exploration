cmake_minimum_required(VERSION 3.2)
project(gstorm)

set(CMAKE_MODULE_PATH ${COMPUTECPP_PACKAGE_ROOT_DIR}/sdk/cmake/Modules)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall")

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)


add_compile_options(-std=c++14 -stdlib=libc++)
add_definitions(${LLVM_DEFINITIONS})

include_directories("${RANGES_V3_PATH}/include")
include_directories("home/m_haid02/projects/computeCpp/include")
include_directories("./include")

set(SOURCE_FILES tests/transform.cpp)
add_executable(gstorm ${SOURCE_FILES})
add_sycl_to_target(gstorm ${CMAKE_CURRENT_BINARY_DIR} ${SOURCE_FILES})