# Link specific libraries

add_executable(RaytraceBenchmark RaytraceBenchmark.cpp)
target_link_libraries(RaytraceBenchmark PRIVATE VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda VecGeom::vgdml)
