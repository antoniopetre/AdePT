// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"

#include <AdePT/BlockData.h>
#include <AdePT/SparseArray.h>

#include <VecGeom/base/Global.h>

inline namespace COPCORE_IMPL {

using RayBlock = adept::BlockData<Ray_t>;
using Array_t        = adept::SparseArray<Ray_t, 1<<20>;

// For the first generation, "create" the rays
__host__ __device__
void generateRays(int id, const RaytracerData_t &rtdata, NavIndex_t *input_buffer)
{
  // Check if the index is greater than the size of the matrix
  if (id > rtdata.fSize_px*rtdata.fSize_py) return;

  Ray_t *ray = (Ray_t *)(input_buffer + id * sizeof(Ray_t));
  ray->index = id;

  // Add the ray in the ray container
  rtdata.sparse_rays[0]->next_free(*ray);
}

COPCORE_CALLABLE_FUNC(generateRays)

__host__ __device__
void renderKernels(int id, const RaytracerData_t &rtdata, NavIndex_t *output_buffer, int generation)
{
  // Propagate all rays and write out the image on the backend

  // Check if the slot is used
  if (!(rtdata.sparse_rays)[generation]->is_used(id)) return;
  
  int ray_index = (*rtdata.sparse_rays[generation])[id].index;

  int px = 0;
  int py = 0;

  if (ray_index) {
    px = ray_index % rtdata.fSize_px;
    py = ray_index / rtdata.fSize_px;
  }
  
  if ((px >= rtdata.fSize_px) || (py >= rtdata.fSize_py)) return;

  auto pixel_color = Raytracer::RaytraceOne(rtdata, px, py, id, generation);

  int pixel_index = 4 * ray_index;
  output_buffer[pixel_index + 0] += pixel_color.fComp.red;
  output_buffer[pixel_index + 1] += pixel_color.fComp.green;
  output_buffer[pixel_index + 2] += pixel_color.fComp.blue;
  output_buffer[pixel_index + 3] = 255;
  
}
COPCORE_CALLABLE_FUNC(renderKernels)


__host__ __device__ void print_rays(const RaytracerData_t &rtdata, int generation)
{
  int px = rtdata.fSize_px;
  int py = rtdata.fSize_py;
  adept::SparseArray<Ray_t, 1<<20> **rays_containers = rtdata.sparse_rays;
  printf(" data: ");
  for (int i = 0; i < px*py; ++i) {
    printf(" %d", (*rays_containers)[generation][i].index);
    if (!(*rays_containers)[generation].is_used(i)) printf("x");
  }
  printf("...\n");
}

__host__ __device__ void print_array(Array_t *array)
{
  printf("=== array: fNshared=%d/%d fNused=%d fNselected=%d - shared=%.1f%% sparsity=%.1f%% selected=%.1f%%\n\n",
         (int)array->size_shared(), (int)array->size_max(), (int)array->size_used(), array->size_selected(),
         100. * array->get_shared_fraction(), 100. * array->get_sparsity(), 100. * array->get_selected_fraction());
}

} // End namespace COPCORE_IMPL
