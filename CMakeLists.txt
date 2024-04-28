# ==================================================
# = Minimum CMake and Project Setup
# ==================================================
cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(strgpt LANGUAGES CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "Please use Clang to build this project")
endif()

# ==================================================
# = Compilation Settings
# ==================================================
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 20)

# ==================================================
# = Dependencies Setup
# ==================================================
include(FetchContent)

# ------------------- libtorch ---------------------
# Check if building on MacOS and adjust the URL accordingly
if(APPLE)
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.0.zip")
else()
  set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcpu.zip")
endif()

FetchContent_Declare(libtorch URL ${LIBTORCH_URL})
FetchContent_MakeAvailable(libtorch)

list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ------------------- re2 ---------------------
FetchContent_Declare(
  re2
  GIT_REPOSITORY https://github.com/google/re2.git
  GIT_TAG b84e3ff189980a33d4a0c6fa1201aa0b3b8bab4a
)
FetchContent_MakeAvailable(re2)

# ------------------- Google Test ---------------------
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# ==================================================
# = Executable Definitions
# ==================================================

# ------------------- Main Binary ---------------------
add_executable(${PROJECT_NAME}
  src/main.cc
  src/bpe.h
  src/bpe.cc
)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}" re2::re2)

# ------------------- Tests ---------------------
enable_testing()
add_executable(${PROJECT_NAME}_tests
  src/bpe_test.cc
  src/bpe.h
  src/bpe.cc
)
target_link_libraries(${PROJECT_NAME}_tests "${TORCH_LIBRARIES}" re2::re2 GTest::gtest_main)
target_compile_definitions(${PROJECT_NAME}_tests PRIVATE UNIT_TEST)
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_tests)

# ==================================================
# = Apple Specific Configuration
# ==================================================
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
  target_link_libraries(${PROJECT_NAME}_tests "${LIBOMP_LIB_PATH}")

  # Setting the RPATH so that libomp.dylib and other dynamic libraries can be found at runtime
  set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
  set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${LIBOMP_PREFIX}/lib;${TORCH_INSTALL_PREFIX}/lib")

  set_target_properties(${PROJECT_NAME}_tests PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
  set_target_properties(${PROJECT_NAME}_tests PROPERTIES INSTALL_RPATH "${LIBOMP_PREFIX}/lib;${TORCH_INSTALL_PREFIX}/lib")
endif()

