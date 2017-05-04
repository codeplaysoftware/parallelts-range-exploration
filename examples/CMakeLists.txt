cmake_minimum_required(VERSION 3.5)

project(sycl-rangeTS)
set(CMAKE_MODULE_PATH ${COMPUTECPP_PACKAGE_ROOT_DIR}/sdk/cmake/Modules)
set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pedantic -Wall")

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)

file(GLOB _ALL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_FILE ${_ALL_FILES})
  if (IS_DIRECTORY ${_FILE})
      if (EXISTS ${_FILE}/CMakeLists.txt)
          add_subdirectory(${_FILE})
      endif()
  endif()
endforeach()
