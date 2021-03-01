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


  // if (ray_index == 61163)
  // {
  //   printf("before white\n");
  //   pixel_color.print();
  //   color[ray_index].print();

  //   adept::Color_t x = 0;
  //   adept::Color_t y = x + pixel_color;
  //   y.print();
    
  // }

  // if (ray_index == 197744) {
  //   printf("before blue\n");
  //   pixel_color.print();
  //   color[ray_index].print();

  //   adept::Color_t x = 0;
  //   adept::Color_t y = x + pixel_color;
  //   y.print();

  // }


  if (ray_index == 179753) {
    // pixel_color.fComp.red = 254;
    // pixel_color.fComp.blue = 254;
    // pixel_color.fComp.green = 254;

    printf("before black\n");
    pixel_color.print();
    color[ray_index].print();

    adept::Color_t x = 0;
    adept::Color_t y = x + pixel_color;
    y.print();
  }
  

  color[ray_index] += pixel_color;
  

  if (ray_index == 179753) {
    printf("after black\n");
    color[ray_index].print();

    adept::Color_t medeea = 0;
    // medeea.print();
    medeea = rtdata.fBkgColor;
    medeea.fComp.alpha = 0;
    medeea.print();

    adept::Color_t x = 0;
    x.fComp.alpha = 255;
    x += medeea;
    x.print();
  }
  // if (ray_index == 197744) {
  //   printf("after blue\n");
  //   color[ray_index].print();
  // }
  // if (ray_index == 61163) {
  //   printf("after white\n");
  //   color[ray_index].print();
  // }
  
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