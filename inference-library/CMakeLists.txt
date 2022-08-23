cmake_minimum_required(VERSION 2.8.12)

project(intermittent-cnn)

set(MY_DEBUG "1" CACHE STRING "Local debug flag. See my_debug.h for details.")
option(ENABLE_COVERAGE "Enable coverage via gcov." OFF)
option(USE_PROTOBUF "Use Protobuf to save results" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Too many parameters are unused if some features are not enabled
set(COMMON_FLAGS "-Wall -Wextra -Wshadow -Wundef -Wno-unused-parameter -pedantic")
if (${MY_DEBUG} GREATER 1)
    set(COMMON_FLAGS "${COMMON_FLAGS} -g -O0")
else ()
    set(COMMON_FLAGS "${COMMON_FLAGS} -O3")
endif ()
if (ENABLE_COVERAGE)
    set(COMMON_FLAGS "${COMMON_FLAGS} --coverage -fprofile-dir=gcov/%p")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage")
endif ()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} -Wstrict-prototypes -std=c11")
# Don't use something that require C++17 or newer, which are not supported by MSP-CGT yet
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} -std=c++14 -Wold-style-cast")

if (USE_PROTOBUF)
    find_package(Protobuf REQUIRED)
    protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS model_output.proto)
    protobuf_generate_python(PROTO_PY model_output.proto)
    add_custom_target(model_output_pb2 ALL DEPENDS ${PROTO_PY})
endif ()

set(COMMON_SRC_PATH ${CMAKE_CURRENT_SOURCE_DIR}/common)

set(DSPLIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/TI-DSPLib)
add_library(dsplib
    ${DSPLIB_PATH}/source/matrix/msp_matrix_mpy_q15.c
    ${DSPLIB_PATH}/source/vector/msp_add_q15.c
    ${DSPLIB_PATH}/source/vector/msp_mpy_q15.c
    ${DSPLIB_PATH}/source/vector/msp_sub_q15.c
    ${DSPLIB_PATH}/source/vector/msp_offset_q15.c
    ${DSPLIB_PATH}/source/vector/msp_max_q15.c
    ${DSPLIB_PATH}/source/vector/msp_min_q15.c
    ${DSPLIB_PATH}/source/vector/msp_scale_q15.c
    ${DSPLIB_PATH}/source/utility/msp_deinterleave_q15.c
    ${DSPLIB_PATH}/source/utility/msp_interleave_q15.c
    ${DSPLIB_PATH}/source/utility/msp_fill_q15.c
)
target_include_directories(dsplib
    SYSTEM PUBLIC
        ${DSPLIB_PATH}/include
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/fake-msp430sdk
    PRIVATE
        ${COMMON_SRC_PATH}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_BINARY_DIR}
)

set (intermittent_cnn_SOURCES
    ${COMMON_SRC_PATH}/intermittent-cnn.cpp
    ${COMMON_SRC_PATH}/op_handlers.cpp
    ${COMMON_SRC_PATH}/op_utils.cpp
    ${COMMON_SRC_PATH}/conv.cpp
    ${COMMON_SRC_PATH}/gemm.cpp
    ${COMMON_SRC_PATH}/pooling.cpp
    ${COMMON_SRC_PATH}/cnn_common.cpp
    ${COMMON_SRC_PATH}/my_debug.cpp
    ${COMMON_SRC_PATH}/plat-linux.cpp
    ${COMMON_SRC_PATH}/platform.cpp
    ${COMMON_SRC_PATH}/my_dsplib.cpp
    ${CMAKE_BINARY_DIR}/data.cpp
)
if (USE_PROTOBUF)
    list(APPEND intermittent_cnn_SOURCES ${PROTO_SRCS})
endif ()
add_executable(intermittent-cnn ${intermittent_cnn_SOURCES})

target_compile_definitions(intermittent-cnn
    PRIVATE
        POSIX_BUILD
        MY_DEBUG=${MY_DEBUG}
)

target_include_directories(intermittent-cnn
    PRIVATE
        ${COMMON_SRC_PATH}
        ${CMAKE_BINARY_DIR}
)

target_link_libraries(intermittent-cnn dsplib)

if (USE_PROTOBUF)
    target_compile_definitions(intermittent-cnn
        PRIVATE
            USE_PROTOBUF
    )
    target_include_directories(intermittent-cnn
        PRIVATE ${CMAKE_CURRENT_BINARY_DIR}  # for *.pb.h files
    )
    target_link_libraries(intermittent-cnn protobuf::libprotobuf)
endif ()
