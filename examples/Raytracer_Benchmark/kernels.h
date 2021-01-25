// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "Raytracer.h"

#include <AdePT/BlockData.h>

#include <VecGeom/base/Global.h>

inline namespace COPCORE_IMPL {

using RayBlock = adept::BlockData<Ray_t>;

// Alocate slots for the BlockData
__host__ __device__
void generateRays(int id, adept::BlockData<Ray_t> *rays)
{
  auto ray = rays->NextElement();
  if (!ray) COPCORE_EXCEPTION("generateRays: Not enough space for rays");

  ray->index = -1;
}
COPCORE_CALLABLE_FUNC(generateRays)

// Initialize the first generation of rays
__host__ __device__
void initialRays(int id, adept::BlockData<Ray_t> *rays, NavIndex_t *input_buffer)
{
  Ray_t *ray = (Ray_t *)(input_buffer + id * sizeof(Ray_t));
  ray->index = id;

  (*rays)[id] = *ray;
}
COPCORE_CALLABLE_FUNC(initialRays)

__host__ __device__
void renderKernels(int id, const RaytracerData_t &rtdata, NavIndex_t *output_buffer, int generation)
{
  // Propagate all rays and write out the image on the backend

  adept::BlockData<RayBlock *> *rays_containers = rtdata.rays;
  RayBlock *rays                                = (*rays_containers)[generation];

  // Check if the slot is used
  if ((*rays)[id].index == -1) return;

  // Get the index of the pixel
  int ray_index = (*rays)[id].index;

  int px = 0;
  int py = 0;

  if (ray_index) {
    px = ray_index % rtdata.fSize_px;
    py = ray_index / rtdata.fSize_px;
  }

  if ((px >= rtdata.fSize_px) || (py >= rtdata.fSize_py)) return;

  auto pixel_color = Raytracer::RaytraceOne(rtdata, rays, px, py, id);

  int pixel_index = 4 * ray_index;
  output_buffer[pixel_index + 0] += pixel_color.fComp.red;
  output_buffer[pixel_index + 1] += pixel_color.fComp.green;
  output_buffer[pixel_index + 2] += pixel_color.fComp.blue;
  output_buffer[pixel_index + 3] = 255;
}
COPCORE_CALLABLE_FUNC(renderKernels)

} // End namespace COPCORE_IMPL
