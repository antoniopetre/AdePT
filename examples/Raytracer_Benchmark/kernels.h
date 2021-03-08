// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"

#include <AdePT/BlockData.h>
#include <AdePT/SparseVector.h>

#include <VecGeom/base/Global.h>

inline namespace COPCORE_IMPL {

// Add initial rays in SparseVector
__host__ __device__ void generateRays(int id, const RaytracerData_t &rtdata)
{
  Ray_t *ray = rtdata.sparse_rays[0]->next_free();
  ray->index = id;
  ray->generation = 0;
  ray->intensity  = 1.;

  Raytracer::InitRay(rtdata, *ray);
}

COPCORE_CALLABLE_FUNC(generateRays)

__host__ __device__ void renderKernels(int id, const RaytracerData_t &rtdata, int generation, adept::Color_t *color)
{
  // Propagate all rays and write out the image on the backend
  if (!(rtdata.sparse_rays)[generation]->is_used(id)) return;
    
  Ray_t &ray = (*rtdata.sparse_rays[generation])[id];

  auto pixel_color = Raytracer::RaytraceOne(rtdata, ray, generation);
  color[ray.index] += pixel_color;
}
COPCORE_CALLABLE_FUNC(renderKernels)

__host__ __device__ void print_vector(adept::SparseVector<Ray_t, 1<<22> *vect)
{
  printf("=== vect: fNshared=%lu/%lu fNused=%lu fNbooked=%lu - shared=%.1f%% sparsity=%.1f%%\n", vect->size(),
         vect->capacity(), vect->size_used(), vect->size_booked(), 100. * vect->get_shared_fraction(),
         100. * vect->get_sparsity());
}

// Check if there are rays in containers
__host__ __device__ bool check_used(const RaytracerData_t &rtdata, int no_generations)
{
  
  auto rays_containers = rtdata.sparse_rays;

  for (int i = 0; i < no_generations; ++i) {
    if (rays_containers[i]->size_used() > 0)
      return true;
  }

  return false;
}

} // End namespace COPCORE_IMPL