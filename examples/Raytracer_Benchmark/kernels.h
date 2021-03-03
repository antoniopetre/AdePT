// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"

#include <AdePT/BlockData.h>
#include <AdePT/SparseVector.h>

#include <VecGeom/base/Global.h>

inline namespace COPCORE_IMPL {

// Add initial rays in SparseVector
__host__ __device__ void generateRays(int id, const RaytracerData_t &rtdata, NavIndex_t *input_buffer)
{
  int ray_index = id;

  Ray_t *ray = (Ray_t *)(input_buffer + id * sizeof(Ray_t));
  ray->index      = ray_index;
  ray->generation = 0;
  ray->intensity  = 1.;

  rtdata.sparse_rays[0]->next_free(*ray);
}

COPCORE_CALLABLE_FUNC(generateRays)


__host__ __device__ void renderKernels(int id, const RaytracerData_t &rtdata, int generation, adept::Color_t *color)
{
  // Propagate all rays and write out the image on the backend
  // size_t n10  = 0.1 * rtdata.fNrays;
  int ray_index;
  int px = 0, py = 0;

  // Propagate the rays
  if (!(rtdata.sparse_rays)[generation]->is_used(id)) return;

  ray_index = (*rtdata.sparse_rays[generation])[id].index;
    
  if (ray_index) {
    px = ray_index % rtdata.fSize_px;
    py = ray_index / rtdata.fSize_px;
  }

  Ray_t *ray = &(*rtdata.sparse_rays[generation])[id];

  auto pixel_color = Raytracer::RaytraceOne(rtdata, *ray, px, py, generation);  
  color[ray_index] += pixel_color;

}
COPCORE_CALLABLE_FUNC(renderKernels)

__host__ __device__ void print_vector(adept::SparseVector<Ray_t, 1<<20> *vect)
{
  printf("=== vect: fNshared=%lu/%lu fNused=%lu fNbooked=%lu - shared=%.1f%% sparsity=%.1f%%\n", vect->size(),
         vect->capacity(), vect->size_used(), vect->size_booked(), 100. * vect->get_shared_fraction(),
         100. * vect->get_sparsity());
}

// Check if there are rays in containers
__host__ __device__ bool check_used(const RaytracerData_t &rtdata, int no_generations)
{
  
  adept::SparseVector<Ray_t, 1<<20> **rays_containers = rtdata.sparse_rays;

  for (int i = 0; i < no_generations; ++i) {
    if (rays_containers[i]->size_used() > 0)
      return true;
  }

  return false;
}

} // End namespace COPCORE_IMPL