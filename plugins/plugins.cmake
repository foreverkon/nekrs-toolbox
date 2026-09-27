find_package(ADIOS2 CONFIG REQUIRED COMPONENTS CXX MPI HINTS "${NEKRS_INSTALL_DIR}")
set(TOOLBOX_DIR "${CMAKE_CURRENT_LIST_DIR}")
target_sources(udf PRIVATE
  "${TOOLBOX_DIR}/common.cpp"
  "${TOOLBOX_DIR}/toolbox.cpp"
  "${TOOLBOX_DIR}/alg.cpp"
  "${TOOLBOX_DIR}/mlog.cpp"
  "${TOOLBOX_DIR}/monitor.cpp"
  "${TOOLBOX_DIR}/probe_io.cpp"
  "${TOOLBOX_DIR}/diagnostics.cpp"
  "${TOOLBOX_DIR}/statistics.cpp"
  "${TOOLBOX_DIR}/noise.cpp")
target_include_directories(udf PRIVATE "${TOOLBOX_DIR}")
target_compile_definitions(udf PRIVATE NEKRS_TOOLBOX_KERNEL_DIR="${TOOLBOX_DIR}/kernels")
target_link_libraries(udf PRIVATE adios2::cxx_mpi)
