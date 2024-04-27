cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(strgpt LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "Please use Clang to build this project")
endif()

# Check if building on MacOS and adjust the URL accordingly
if(APPLE)
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.0.zip")
else()
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip")
endif()

include(FetchContent)
FetchContent_Declare(libtorch URL ${LIBTORCH_URL})

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
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")

if(APPLE)
  # Execute process to get the path to libomp
  execute_process(
    COMMAND brew --prefix libomp
    OUTPUT_VARIABLE LIBOMP_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(LIBOMP_LIB_PATH "${LIBOMP_PREFIX}/lib/libomp.dylib")
  if(NOT EXISTS "${LIBOMP_LIB_PATH}")
    message(FATAL_ERROR "libomp.dylib not found at ${LIBOMP_LIB_PATH}. Please ensure it is installed.")
  endif()
  target_link_libraries(${PROJECT_NAME} "${LIBOMP_LIB_PATH}")

  # Setting the RPATH so that libomp.dylib and other dynamic libraries can be found at runtime
  set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
  set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${LIBOMP_PREFIX}/lib;${TORCH_INSTALL_PREFIX}/lib")
endif()

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 20)
