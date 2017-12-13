cmake_minimum_required(VERSION 3.5)
project(tryout)

include_directories(${COMPUTECPP_INCLUDE_DIRECTORY} ${CMAKE_CURRENT_SOURCE_DIR}/../../gstorm/include ${CMAKE_CURRENT_SOURCE_DIR}/../../range-v3/include)

add_compile_options(-std=c++17)

# Device compiler doesn't support '-std=c++17'
# Define NDEBUG to avoid <cassert> functions in device code
set(COMPUTECPP_DEVICE_COMPILER_FLAGS "${COMPUTECPP_DEVICE_COMPILER_FLAGS} -std=c++1z -DNDEBUG")

set(SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/${PROJECT_NAME}.cpp)

function(add_app name)
  add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
  add_sycl_to_target(${name} ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${name}.cpp)
endfunction()

add_app(accessor)
add_app(transform)
add_app(view_zip_workaround_ptr)
add_app(view_zip_workaround_copy)
add_app(my_zip_just_gvector)
add_app(my_zip_gvector_view)
add_app(view_transform)
add_app(view_transform_2)
add_app(transform_non_multiple)

add_app(sum)
add_app(saxpy)
add_app(dot_product)
add_app(dot_product_zip)
add_app(dot_product_zip_copy)
# add_app(dot_product_slow)

add_app(view_tail)
add_app(view_take)
add_app(view_drop)
add_app(view_ints)
add_app(view_slice)
add_app(view_intersperse)
