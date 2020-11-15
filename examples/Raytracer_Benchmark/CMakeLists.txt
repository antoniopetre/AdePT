# Link specific libraries

# library Raytracer.cu
cuda_add_library(Raytracer_cuda Raytracer.cu)
target_link_libraries(Raytracer_cuda VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda VecGeom::vgdml)
target_compile_options(Raytracer_cuda PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;--extended-lambda;--device-c;-lcudart;-lcudadevrt>")

# library RaytraceBenchmark.cu Raytracer.cpp
add_library(Raytracer Raytracer.cpp RaytraceBenchmark.cu)
target_link_libraries(Raytracer VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda VecGeom::vgdml Raytracer_cuda)
target_compile_options(Raytracer PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;--extended-lambda;--device-c;-lcudart;-lcudadevrt>")


#library RaytraceBenchmark.cpp
add_executable(RaytraceBenchmark RaytraceBenchmark.cpp)
target_link_libraries(RaytraceBenchmark VecCore::VecCore VecGeom::vecgeom VecGeom::vecgeomcuda VecGeom::vgdml Raytracer Raytracer_cuda)
target_compile_options(RaytraceBenchmark PRIVATE "$<$<AND:$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx;--extended-lambda;--device-c;-lcudart;-lcudadevrt>")
