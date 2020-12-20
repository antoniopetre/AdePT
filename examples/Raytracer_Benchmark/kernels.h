// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"

#include <AdePT/BlockData.h>
#include <AdePT/MParray.h>

#include <VecGeom/base/Global.h>

inline namespace COPCORE_IMPL {

// Alocate slots for the BlockData
VECCORE_ATT_HOST_DEVICE
void generateRays(int id, adept::BlockData<Ray_t> *rays)
{
  auto ray = rays->NextElement();
  if (!ray) COPCORE_EXCEPTION("generateRays: Not enough space for rays");

  ray->index = id;
}

COPCORE_CALLABLE_FUNC(generateRays)

VECCORE_ATT_HOST_DEVICE
void renderKernels(int id, adept::BlockData<Ray_t> *rays, adept::BlockData<Ray_t> *secondary_rays, const RaytracerData_t &rtdata, NavIndex_t *input_buffer,
                   NavIndex_t *output_buffer, int time, bool reflected, adept::MParray *pixel_indices)
{
  // Propagate all rays and write out the image on the backend
  // size_t n10  = 0.1 * rtdata.fNrays;

  int ray_index = id;

  ray_index += time*10000;

  int px = 0;
  int py = 0;

  if (ray_index) {
    px = ray_index % rtdata.fSize_px;
    py = ray_index / rtdata.fSize_px;
  }

  if ((px >= rtdata.fSize_px) || (py >= rtdata.fSize_py)) return;

  Ray_t *ray, reflected_ray;
    
  if (!reflected) {
    // fprintf(stderr, "P3\n%d %d\n255\n", fSize_px, fSize_py);
    // if ((ray_index % n10) == 0) printf("%lu %%\n", 10 * ray_index / n10);
    ray = (Ray_t *)(input_buffer + ray_index * sizeof(Ray_t));
    ray->index = id;
    reflected_ray = *ray;
    reflected_ray.intensity = 0;

    (*rays)[id] = *ray;
    (*secondary_rays)[id] = reflected_ray;
   
  }

  auto pixel_color = Raytracer::RaytraceOne(rtdata, rays, secondary_rays, px, py, id, time, pixel_indices);

  int pixel_index                 = 4 * ray_index;
  output_buffer[pixel_index + 0] += pixel_color.fComp.red;
  output_buffer[pixel_index + 1] += pixel_color.fComp.green;
  output_buffer[pixel_index + 2] += pixel_color.fComp.blue;
  output_buffer[pixel_index + 3]  = 255;

  if(rtdata.fModel == kRTfresnel) {
    for (int i = 0; i < (*rays)[id].secondary_rays; ++i)
    {
      pixel_color = Raytracer::RaytraceOne(rtdata, secondary_rays, rays, px, py, (*rays)[id].rays[i], time, pixel_indices);
      output_buffer[pixel_index + 0] += pixel_color.fComp.red;
      output_buffer[pixel_index + 1] += pixel_color.fComp.green;
      output_buffer[pixel_index + 2] += pixel_color.fComp.blue;
    }
  }


}
COPCORE_CALLABLE_FUNC(renderKernels)

} // End namespace COPCORE_IMPL
