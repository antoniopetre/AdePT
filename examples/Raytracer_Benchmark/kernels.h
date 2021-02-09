// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"

#include <AdePT/BlockData.h>
#include <AdePT/SparseVector.h>

#include <VecGeom/base/Global.h>

inline namespace COPCORE_IMPL {

// Alocate slots for the BlockData
__host__ __device__
void generateRays(int id, adept::BlockData<Ray_t> *rays)
{
  auto ray = rays->NextElement();
  if (!ray) COPCORE_EXCEPTION("generateRays: Not enough space for rays");

  ray->index = id;
}

COPCORE_CALLABLE_FUNC(generateRays)

__host__ __device__
void renderKernels(int id, const RaytracerData_t &rtdata, NavIndex_t *input_buffer,
                   NavIndex_t *output_buffer, int generation)
{
  // Propagate all rays and write out the image on the backend
  // size_t n10  = 0.1 * rtdata.fNrays;
  int ray_index;
  int px = 0, py = 0;

  if (generation == 0 && id < rtdata.fSize_px*rtdata.fSize_py) {
    // Create the rays
    ray_index = id;

    if (ray_index) {
      px = ray_index % rtdata.fSize_px;
      py = ray_index / rtdata.fSize_px;
    }
    Ray_t *ray = (Ray_t *)(input_buffer + id * sizeof(Ray_t));
    ray->index      = id;
    ray->generation = 0;

    ray = rtdata.sparse_rays[0]->next_free(*ray);

    auto pixel_color = Raytracer::RaytraceOne(rtdata, *ray, px, py, id, generation);

    int pixel_index = 4 * ray_index;
    output_buffer[pixel_index + 0] += pixel_color.fComp.red;
    output_buffer[pixel_index + 1] += pixel_color.fComp.green;
    output_buffer[pixel_index + 2] += pixel_color.fComp.blue;
    output_buffer[pixel_index + 3] = 255;
  }

  else {
    // Propagate the secondary rays
    if (!(rtdata.sparse_rays)[generation]->is_used(id)) return;

    ray_index = (*rtdata.sparse_rays[generation])[id].index;
    
    if (ray_index) {
      px = ray_index % rtdata.fSize_px;
      py = ray_index / rtdata.fSize_px;
    }

    Ray_t ray = (*rtdata.sparse_rays[generation])[id];

    auto pixel_color = Raytracer::RaytraceOne(rtdata, ray, px, py, id, generation);

    int pixel_index = 4 * ray_index;
    output_buffer[pixel_index + 0] += pixel_color.fComp.red;
    output_buffer[pixel_index + 1] += pixel_color.fComp.green;
    output_buffer[pixel_index + 2] += pixel_color.fComp.blue;
    output_buffer[pixel_index + 3] = 255;


  }
  
}
COPCORE_CALLABLE_FUNC(renderKernels)

__host__ __device__ void print_vector(adept::SparseVector<Ray_t, 1<<20> *vect)
{
  printf("=== vect: fNshared=%lu/%lu fNused=%lu fNbooked=%lu - shared=%.1f%% sparsity=%.1f%%\n", vect->size(),
         vect->capacity(), vect->size_used(), vect->size_booked(), 100. * vect->get_shared_fraction(),
         100. * vect->get_sparsity());
}

} // End namespace COPCORE_IMPL
