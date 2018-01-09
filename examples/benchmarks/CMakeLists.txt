function(add_benchmark filename)

  get_filename_component(SOURCE_NAME ${filename} NAME_WE)

  add_executable(${SOURCE_NAME} ${SOURCE_NAME}.cpp)
  add_sycl_to_target(${SOURCE_NAME} ${CMAKE_CURRENT_BINARY_DIR}
                                    ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp)
endfunction()


file(GLOB SRC_FILES "*.cpp")

foreach(file ${SRC_FILES})
  add_benchmark(${file})
endforeach()
