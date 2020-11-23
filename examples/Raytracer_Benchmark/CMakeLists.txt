# Link specific libraries

#exec RaytraceBenchmark.cpp
add_executable(RaytraceBenchmark Raytracer.cpp RaytraceBenchmark.cpp)
target_include_directories(RaytraceBenchmark PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/base/inc>
  $<INSTALL_INTERFACE:base>
)
target_link_libraries(RaytraceBenchmark VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda VecGeom::vgdml raytracercu)
target_compile_options(RaytraceBenchmark PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
set_target_properties(RaytraceBenchmark PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_RESOLVE_DEVICE_SYMBOLS ON)

