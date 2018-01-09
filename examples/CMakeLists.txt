cmake_minimum_required(VERSION 3.5)

project(sycl-rangeTS)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -pedantic -Wall")

# Check that a supported host compiler can be found
if(CMAKE_COMPILER_IS_GNUCXX)
    # Require at least gcc 6.2
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 6.2)
      message(FATAL_ERROR
        "gcc version must be at least 6.2")
    endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    # Require at least clang 3.9
    if (${CMAKE_CXX_COMPILER_VERSION} VERSION_LESS 3.9)
      message(FATAL_ERROR
        "clang version must be at least 3.9")
    endif()
elseif (MSVC)
  message(FATAL_ERROR, "MSVC not supported")
endif()

find_package(OpenCL REQUIRED)
find_package(ComputeCpp REQUIRED)

include_directories("include")
include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../gstorm/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../range-v3/include)

add_compile_options(-std=c++17)

# Device compiler doesn't support '-std=c++17'
# Define NDEBUG to avoid <cassert> functions in device code
set(COMPUTECPP_DEVICE_COMPILER_FLAGS "${COMPUTECPP_DEVICE_COMPILER_FLAGS} -std=c++1z -DNDEBUG -no-serial-memop")

enable_testing()

file(GLOB _ALL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(_FILE ${_ALL_FILES})
  if (IS_DIRECTORY ${_FILE})
      if (EXISTS ${_FILE}/CMakeLists.txt)
          add_subdirectory(${_FILE})
      endif()
  endif()
endforeach()
