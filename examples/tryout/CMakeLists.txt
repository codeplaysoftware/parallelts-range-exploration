cmake_minimum_required(VERSION 3.5)
project(tryout)

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/../../gstorm/include ${CMAKE_CURRENT_SOURCE_DIR}/../../range-v3/include)

add_compile_options(-std=c++1z -fno-exceptions)

set(COMPUTECPP_DEVICE_COMPILER_FLAGS "${COMPUTECPP_DEVICE_COMPILER_FLAGS} -O3 -std=c++1z -fno-exceptions")

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.cpp)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

add_sycl_to_target(${PROJECT_NAME} ${CMAKE_CURRENT_BINARY_DIR} ${SOURCE_FILES})

