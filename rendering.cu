#include <stdint.h>

#include "rendering.h"

// __global__ void kernelDrawPixels(void *memory, int width, int height, double xOffset, double yOffset)
// {
//     int x = threadIdx.x;
//     int y = blockIdx.x;

//     uint8_t red, green, blue;

//     red = x - (int8_t)xOffset;
//     green = y - (int8_t)yOffset;
//     blue = (int8_t)xOffset + (y * (int8_t)yOffset);

//     if (!((x + (int8_t)xOffset) % 256) || !((y + (int8_t)yOffset) % 256))
//     {
//         red = 255;
//         green = 255;
//         blue = 255;
//     }

//     uint32_t *pixel = (uint32_t *)memory;
//     pixel += y * width + x;

//     *pixel = (red << 16) | (green << 8) | (blue << 0);
// }

__global__ void kernelTransform( double *transformation, VERTEX *vertices, VERTEX *transformedVertices)
{
    VERTEX vertex = vertices[threadIdx.x];
    VERTEX transformedVertex;

    /* Just your average matrix multiplication */
    transformedVertex.x = (transformation[0] * vertex.x) + (transformation[1] * vertex.y) + (transformation[2] * vertex.z) + transformation[3];
    transformedVertex.y = (transformation[4] * vertex.x) + (transformation[5] * vertex.y) + (transformation[6] * vertex.z) + transformation[7];
    transformedVertex.z = (transformation[8] * vertex.x) + (transformation[9] * vertex.y) + (transformation[10] * vertex.z) + transformation[11];

    transformedVertices[threadIdx.x] = transformedVertex;
}