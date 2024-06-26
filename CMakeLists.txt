# ==================================================
# = Minimum CMake and Project Setup
# ==================================================
cmake_minimum_required(VERSION 3.27 FATAL_ERROR)
project(strgpt LANGUAGES CXX)

# ==================================================
# = Compilation Settings
# ==================================================
if (NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR "This project requires Clang. GCC is not supported.")
endif ()

# Find the full path to clang and clang++
execute_process(COMMAND which clang OUTPUT_VARIABLE CLANG_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND which clang++ OUTPUT_VARIABLE CLANGXX_PATH OUTPUT_STRIP_TRAILING_WHITESPACE)

# Set the compile flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdangling-gsl -Wreturn-stack-address")

# ==================================================
# = Dependencies Setup
# ==================================================
include(FetchContent)

# ------------------- libtorch ---------------------
if (APPLE) # MacOS for ARM64
    message(STATUS "Downloading libtorch for MacOS")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-2.3.0.zip")
elseif (DEFINED ENV{CUDA_HOME}) # Linux with CUDA support
    # CUDA compiler does not support GCC 9 and later versions
    set(CMAKE_C_COMPILER ${CLANG_PATH})
    set(CMAKE_CXX_COMPILER ${CLANGXX_PATH})

    set(TORCH_CUDA_ARCH_LIST "8.0 8.6 8.9 9.0" CACHE STRING "Architectures for CUDA compiler" FORCE)
    set(CMAKE_CUDA_FLAGS "-allow-unsupported-compiler" CACHE STRING "Flags for CUDA compiler" FORCE)

    message(STATUS "Downloading libtorch with CUDA support")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcu121.zip")
else () # Linux without CUDA support
    message(STATUS "Downloading CPU-only version of libtorch")
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.3.0%2Bcpu.zip")
endif ()

FetchContent_Declare(libtorch URL ${LIBTORCH_URL})
FetchContent_MakeAvailable(libtorch)

list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ------------------- JSON ---------------------
message(STATUS "Downloading github.com/nlohmann/json")
FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

file(DOWNLOAD # Download tokenizer config
        "https://huggingface.co/gpt2/raw/main/tokenizer.json"
        "${CMAKE_CURRENT_BINARY_DIR}/tokenizer.json"
        TIMEOUT 60
        STATUS download_status)
list(GET download_status 0 status_code)# Check if the download was successful
if (NOT status_code EQUAL 0)
    message(FATAL_ERROR "Failed to download file: ${download_status}")
endif ()

# ------------------- args ---------------------
message(STATUS "Downloading github.com/Taywee/args")
FetchContent_Declare(
        args
        URL https://github.com/Taywee/args/archive/cc2368ca0d8a962862c96c00fe919e1480050f51.zip
)
FetchContent_MakeAvailable(args)

# ------------------- Google Test ---------------------
message(STATUS "Downloading github.com/google/googletest")
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
        src/command/train.cc
        src/command/sample.cc
        src/command/dataset.cc
        src/tokenizer/char.cc
        src/tokenizer/bpe.cc
        src/model/model.cc
        src/model/train.cc
)
target_include_directories(${PROJECT_NAME} PRIVATE src)
target_link_libraries(${PROJECT_NAME}
        "${TORCH_LIBRARIES}"
        nlohmann_json::nlohmann_json
        taywee::args
)

# ------------------- Tests ---------------------
enable_testing()
add_executable(${PROJECT_NAME}_tests
        src/tokenizer/char_test.cc
        src/tokenizer/char.cc
        src/tokenizer/bpe_test.cc
        src/tokenizer/bpe.cc
)
target_include_directories(${PROJECT_NAME}_tests PRIVATE src)
target_link_libraries(${PROJECT_NAME}_tests
        GTest::gtest_main
        "${TORCH_LIBRARIES}"
        nlohmann_json::nlohmann_json
)
target_compile_definitions(${PROJECT_NAME}_tests PRIVATE UNIT_TEST)
include(GoogleTest)
gtest_discover_tests(${PROJECT_NAME}_tests)

# ==================================================
# = Target Configurations
# ==================================================

# ------------------- Link Time Optimizations ---------------------
include(CheckIPOSupported)
check_ipo_supported(RESULT supported OUTPUT error)
if (supported)
    message(STATUS "IPO / LTO enabled")
    set_target_property(${PROJECT_NAME} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    set_target_property(${PROJECT_NAME}_tests PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
else ()
    message(WARNING "IPO / LTO not supported: <${error}>")
endif ()

# ------------------- Apple Specific Configuration ---------------------
if (APPLE)
    # Execute process to get the path to libomp
    execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )

    set(LIBOMP_LIB_PATH "${LIBOMP_PREFIX}/lib/libomp.dylib")
    if (NOT EXISTS "${LIBOMP_LIB_PATH}")
        message(FATAL_ERROR "libomp.dylib not found at ${LIBOMP_LIB_PATH}. Please ensure it is installed.")
    endif ()
    target_link_libraries(${PROJECT_NAME} "${LIBOMP_LIB_PATH}")
    target_link_libraries(${PROJECT_NAME}_tests "${LIBOMP_LIB_PATH}")

    # Setting the RPATH so that libomp.dylib and other dynamic libraries can be found at runtime
    set_target_properties(${PROJECT_NAME} PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(${PROJECT_NAME} PROPERTIES INSTALL_RPATH "${LIBOMP_PREFIX}/lib;${TORCH_INSTALL_PREFIX}/lib")

    set_target_properties(${PROJECT_NAME}_tests PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(${PROJECT_NAME}_tests PROPERTIES INSTALL_RPATH "${LIBOMP_PREFIX}/lib;${TORCH_INSTALL_PREFIX}/lib")
endif ()
