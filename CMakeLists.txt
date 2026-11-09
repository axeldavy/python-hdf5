cmake_minimum_required(VERSION 3.24)
project(python_hdf5 LANGUAGES C CXX)

include(FetchContent)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_HL_LIB OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_CPP_LIB OFF CACHE BOOL "" FORCE)
set(HDF5_BUILD_FORTRAN OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_THREADSAFE ON CACHE BOOL "" FORCE)
set(HDF5_ENABLE_Z_LIB_SUPPORT OFF CACHE BOOL "" FORCE)
set(HDF5_ENABLE_SZIP_SUPPORT OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    hdf5
    URL https://github.com/HDFGroup/hdf5/archive/refs/tags/2.2.0.tar.gz
    URL_HASH SHA256=5b8d75125ae7b4fef55d3b39e8d3dbfd238cdf63b6518afd67fa69ac80f06542
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(hdf5)

set_target_properties(hdf5-static PROPERTIES C_VISIBILITY_PRESET hidden)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
set(CYTHON_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/_hdf5.cpp")
add_custom_command(
    OUTPUT "${CYTHON_OUTPUT}"
    COMMAND "${Python_EXECUTABLE}" -m cython --cplus -3
            -o "${CYTHON_OUTPUT}" "${CMAKE_CURRENT_SOURCE_DIR}/hdf5/_hdf5.pyx"
    DEPENDS hdf5/_hdf5.pyx hdf5/cpp/file.hpp hdf5/cpp/group.hpp
            hdf5/cpp/attribute.hpp hdf5/cpp/dataset.hpp
            hdf5/cpp/datatype.hpp
    VERBATIM
)
Python_add_library(_hdf5 MODULE WITH_SOABI "${CYTHON_OUTPUT}")
target_compile_features(_hdf5 PRIVATE cxx_std_17)
target_include_directories(_hdf5 PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/hdf5")
target_link_libraries(_hdf5 PRIVATE hdf5-static)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(_hdf5 PRIVATE "LINKER:--exclude-libs,ALL")
endif()
install(
    TARGETS _hdf5
    LIBRARY DESTINATION hdf5 COMPONENT python_hdf5
    RUNTIME DESTINATION hdf5 COMPONENT python_hdf5
)
install(
    FILES "${hdf5_SOURCE_DIR}/LICENSE"
    DESTINATION hdf5/licenses
    COMPONENT python_hdf5
)
