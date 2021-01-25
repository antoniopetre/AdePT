// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"
#include "kernels.h"

#include <CopCore/Global.h>
#include <CopCore/CopCore.h>
#include <CopCore/Launcher.h>
#include <AdePT/ArgParser.h>
#include <AdePT/BlockData.h>
#include <AdePT/LoopNavigator.h>
#include <AdePT/MParray.h>
#include <AdePT/SparseArray.h>
// #include <CopCore/Ranluxpp.h>

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/management/GeoManager.h>
#include <VecGeom/navigation/NavStatePath.h>
#include <VecGeom/base/Stopwatch.h>
#include <VecGeom/management/CudaManager.h>
#include <VecGeom/base/Global.h>

#ifdef VECGEOM_GDML
#include <VecGeom/gdml/Frontend.h>
#endif

void initiliazeCudaWorld(RaytracerData_t *rtdata);

void RenderTiledImage(adept::BlockData<Ray_t> *rays, RaytracerData_t *rtdata, NavIndex_t *output_buffer,
                      int block_size);

template <copcore::BackendType backend>
void InitRTdata(RaytracerData_t *rtdata)
{

  if (backend == copcore::BackendType::CUDA) {
    initiliazeCudaWorld((RaytracerData_t *)rtdata);
  } else {
    vecgeom::NavStateIndex vpstate;
    LoopNavigator::LocatePointIn(rtdata->fWorld, rtdata->fStart, vpstate, true);
    rtdata->fVPstate = vpstate;
  }
}

template <copcore::BackendType backend>
int runSimulation(const vecgeom::cxx::VPlacedVolume *world, int argc, char *argv[])
{

  // image size in pixels
  OPTION_INT(px, 1840);
  OPTION_INT(py, 512);

  // RT model as in { kRTxray = 0, kRTspecular, kRTtransparent, kRTdiffuse };
  OPTION_INT(model, 2);

  // RT view as in { kRTVparallel = 0, kRTVperspective };
  OPTION_INT(view, 1);

  // zoom w.r.t to the default view mode
  OPTION_DOUBLE(zoom, 3.5);

  // Screen position in world coordinates
  OPTION_DOUBLE(screenx, -5000);
  OPTION_DOUBLE(screeny, 0);
  OPTION_DOUBLE(screenz, 0);

  // Up vector (no need to be normalized)
  OPTION_DOUBLE(upx, 0);
  OPTION_DOUBLE(upy, 1);
  OPTION_DOUBLE(upz, 0);
  vecgeom::Vector3D<double> up(upx, upy, upz);

  // Light color, object color (no color per volume yet) - in RGBA chars compressed into an unsigned integer
  OPTION_INT(bkgcol, 0xFF0000FF); // red
  OPTION_INT(objcol, 0x0000FFFF); // blue
  OPTION_INT(vdepth, 4);          // visible depth

  OPTION_INT(use_tiles, 0);  // run on GPU in tiled mode
  OPTION_INT(block_size, 8); // run on GPU in tiled mode

  
  copcore::Allocator<RaytracerData_t, backend> raydataAlloc;
  RaytracerData_t *rtdata = raydataAlloc.allocate(1);

  rtdata->fScreenPos.Set(screenx, screeny, screenz);
  rtdata->fUp.Set(upx, upy, upz);
  rtdata->fZoom     = zoom;
  rtdata->fModel    = (ERTmodel)model;
  rtdata->fView     = (ERTView)view;
  rtdata->fSize_px  = px;
  rtdata->fSize_py  = py;
  rtdata->fBkgColor = bkgcol;
  rtdata->fObjColor = objcol;
  rtdata->fVisDepth = vdepth;

  Raytracer::InitializeModel((Raytracer::VPlacedVolumePtr_t)world, *rtdata);

  InitRTdata<backend>(rtdata);

  rtdata->Print();

  using RayBlock       = adept::BlockData<Ray_t>;
  using RayAllocator   = copcore::VariableSizeObjAllocator<RayBlock, backend>;
  using Block          = adept::BlockData<RayBlock *>;
  using BlockAllocator = copcore::VariableSizeObjAllocator<Block, backend>;
  using Launcher_t     = copcore::Launcher<backend>;
  using StreamStruct   = copcore::StreamType<backend>;
  using Stream_t       = typename StreamStruct::value_type;
  using Array_t        = adept::SparseArray<Ray_t, 1<<20>;
  using LaunchGrid_t = copcore::launch_grid<copcore::BackendType::CUDA>;

  LaunchGrid_t grid = LaunchGrid_t();

  int capacity       = 1 << 20;
  int no_generations = 1;

  if (rtdata->fModel == kRTfresnel) no_generations = 10;

  // Allocate the rays container
  Array_t **array_ptr;
  cudaMallocManaged(&array_ptr, sizeof(Array_t *)); 

  for (int i = 0; i < no_generations; ++i)
  {
    cudaMallocManaged(&(array_ptr[i]), sizeof(Array_t));
    Array_t::MakeInstanceAt(array_ptr[i]);
  }

  cudaDeviceSynchronize();
  rtdata->sparse_rays = array_ptr;

  // Boilerplate to get the pointers to the device functions to be used
  COPCORE_CALLABLE_DECLARE(generateFunc, generateRays);
  COPCORE_CALLABLE_DECLARE(renderkernelFunc, renderKernels);

  // Create a stream to work with.
  Stream_t stream;
  StreamStruct::CreateStream(stream);
  Launcher_t generate(stream);

  // Allocate and initialize all rays on the host
  size_t raysize = Ray_t::SizeOfInstance();
  printf("=== Allocating %.3f MB of ray data on the %s\n", (float)rtdata->fNrays * raysize / 1048576,
         copcore::BackendName(backend));

  copcore::Allocator<NavIndex_t, backend> charAlloc;
  NavIndex_t *input_buffer = charAlloc.allocate(rtdata->fNrays * raysize * sizeof(NavIndex_t));

  copcore::Allocator<NavIndex_t, backend> ucharAlloc;
  NavIndex_t *output_buffer = ucharAlloc.allocate(4 * rtdata->fNrays * sizeof(NavIndex_t));

  // Construct rays in place
  for (int iray = 0; iray < rtdata->fNrays; ++iray)
    Ray_t::MakeInstanceAt(input_buffer + iray * raysize);

  // Initialize the rays container
  generate.Run(generateFunc, capacity, {0, 0}, *rtdata, input_buffer);
  generate.WaitStream();

  vecgeom::Stopwatch timer;
  timer.Start();

  if (backend == copcore::BackendType::CUDA && use_tiles) {
    // RenderTiledImage((*rays)[0], (RaytracerData_t *)rtdata, output_buffer, block_size);
  } else {
    Launcher_t renderKernel(stream);
    while(check_used(*rtdata, no_generations)) {
      for (int i = 0; i < no_generations; i++) {
        // Propagate the rays
        renderKernel.Run(renderkernelFunc, capacity, {0, 0}, *rtdata, output_buffer, i);
        renderKernel.WaitStream();

        // Select the rays from container
        rtdata->sparse_rays[i]->select([] __device__(int i, const Array_t *arr) { return ((*arr)[i].fDone == false); });
        cudaDeviceSynchronize();

        // Release the selected rays 
        rtdata->sparse_rays[i]->release_selected(grid);
        renderKernel.WaitStream();

        rtdata->sparse_rays[i]->select_used();
        renderKernel.WaitStream();
      }
    }
  }

  // Print basic information about containers
  // for (int i = 0; i < no_generations; ++i)
  //   print_array(rtdata->sparse_rays[i]);

  auto time_cpu = timer.Stop();
  std::cout << "Run time: " << time_cpu << "\n";

  // Write the output
  write_ppm("output.ppm", output_buffer, rtdata->fSize_px, rtdata->fSize_py);

  return 0;
}
