# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in
               ${CMAKE_BINARY_DIR}/googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/googletest-download )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory(${CMAKE_BINARY_DIR}/googletest-src
                 ${CMAKE_BINARY_DIR}/googletest-build)
include_directories(${gtest_SOURCE_DIR}/include)

function(add_new_test filename)

  get_filename_component(SOURCE_NAME ${filename} NAME_WE)

  set(TEST_NAME "test_${SOURCE_NAME}")

  add_executable(${TEST_NAME} ${SOURCE_NAME}.cpp)
  target_link_libraries(${TEST_NAME} PUBLIC ${gtest_BINARY_DIR}/libgtest.a
                                     PUBLIC ${gtest_BINARY_DIR}/libgtest_main.a
                                     PUBLIC pthread)
  add_dependencies(${TEST_NAME} gtest_main)
  add_dependencies(${TEST_NAME} gtest)
  add_sycl_to_target(${TEST_NAME} ${CMAKE_CURRENT_BINARY_DIR}
                                  ${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_NAME}.cpp)

  add_test(${TEST_NAME} ${TEST_NAME})
endfunction()


file(GLOB SRC_FILES "*.cpp")

foreach(file ${SRC_FILES})
  add_new_test(${file})
endforeach()
