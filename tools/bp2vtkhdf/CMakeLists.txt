cmake_minimum_required(VERSION 3.21)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
  project(bp2vtkhdf VERSION 0.2.0 LANGUAGES C CXX)
endif()

set(NEKRS_HOME "$ENV{NEKRS_HOME}" CACHE PATH "nekRS installation prefix")

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR
   AND CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT AND NEKRS_HOME)
  set(CMAKE_INSTALL_PREFIX "${NEKRS_HOME}" CACHE PATH "Installation prefix" FORCE)
endif()

function(bp2vtkhdf_find_dependencies)
  if(NEKRS_HOME)
    list(APPEND CMAKE_PREFIX_PATH
      "${NEKRS_HOME}"
      "${NEKRS_HOME}/adios2"
      "${NEKRS_HOME}/hdf5"
      "${NEKRS_HOME}/zlib")
  endif()

  if(NOT TARGET adios2::cxx)
    find_package(ADIOS2 CONFIG REQUIRED)
  endif()
  if(NOT TARGET Threads::Threads)
    find_package(Threads REQUIRED)
  endif()

  include(FetchContent)
  if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
  endif()

  if(NOT TARGET ZLIB::ZLIB)
    find_package(ZLIB QUIET)
    if(NOT ZLIB_FOUND)
      message(STATUS "ZLIB not found; fetching zlib 1.3.1")
      set(ZLIB_BUILD_EXAMPLES OFF)
      FetchContent_Declare(zlib
        URL https://zlib.net/fossils/zlib-1.3.1.tar.gz
        URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23)
      FetchContent_MakeAvailable(zlib)
      add_library(ZLIB::ZLIB ALIAS zlibstatic)
    endif()
  endif()

  if(NOT TARGET HDF5::HDF5)
    find_package(HDF5 QUIET COMPONENTS C)
    if(NOT HDF5_FOUND)
      message(STATUS "HDF5 not found; fetching HDF5 1.14.6")
      set(BUILD_SHARED_LIBS OFF)
      set(BUILD_STATIC_LIBS ON)
      set(BUILD_TESTING OFF)
      set(HDF5_BUILD_TOOLS ON)
      set(HDF5_BUILD_UTILS OFF)
      set(HDF5_BUILD_EXAMPLES OFF)
      set(HDF5_BUILD_HL_LIB OFF)
      set(HDF5_BUILD_CPP_LIB OFF)
      set(HDF5_BUILD_FORTRAN OFF)
      set(HDF5_BUILD_JAVA OFF)
      set(HDF5_ENABLE_PARALLEL OFF)
      set(HDF5_ENABLE_Z_LIB_SUPPORT ON)
      set(HDF5_ENABLE_SZIP_SUPPORT OFF)
      set(HDF5_ALLOW_EXTERNAL_SUPPORT NO)
      set(H5_ZLIB_HEADER "zlib.h")
      set(LINK_COMP_LIBS ZLIB::ZLIB)
      FetchContent_Declare(hdf5
        URL https://codeload.github.com/HDFGroup/hdf5/tar.gz/refs/tags/hdf5_1.14.6
        URL_HASH SHA256=09ee1c671a87401a5201c06106650f62badeea5a3b3941e9b1e2e1e08317357f)
      FetchContent_MakeAvailable(hdf5)
      add_library(HDF5::HDF5 ALIAS hdf5-static)
    endif()
  endif()
endfunction()

bp2vtkhdf_find_dependencies()

add_executable(bp2vtkhdf main.cpp)
target_compile_features(bp2vtkhdf PRIVATE cxx_std_17)
target_link_libraries(bp2vtkhdf PRIVATE
  adios2::cxx HDF5::HDF5 ZLIB::ZLIB Threads::Threads)
if(MSVC)
  target_compile_options(bp2vtkhdf PRIVATE /W4 /permissive- /EHsc)
else()
  target_compile_options(bp2vtkhdf PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(GNUInstallDirs)
install(TARGETS bp2vtkhdf RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT BPTools)
set_target_properties(bp2vtkhdf PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
