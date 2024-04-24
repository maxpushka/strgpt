cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(strgpt LANGUAGES CXX)

include(FetchContent)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Please use Clang to build this project")
endif()

FetchContent_Declare(
  libtorch
  URL https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
)

# Download and unpack LibTorch if it hasn't been done already
FetchContent_GetProperties(libtorch)
if(NOT libtorch_POPULATED)
  FetchContent_Populate(libtorch)
  set(TORCH_INSTALL_PREFIX "${libtorch_SOURCE_DIR}") # Set the path for find_package to use
endif()

# Find LibTorch
list(APPEND CMAKE_PREFIX_PATH "${TORCH_INSTALL_PREFIX}")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET ${PROJECT_NAME}
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:${PROJECT_NAME}>)
endif (MSVC)
