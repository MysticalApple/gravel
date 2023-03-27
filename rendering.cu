#include "rendering.h"
#include <stdint.h>

/* Sets a given pixel to white */
void DrawPoint(win32_offscreen_buffer *buffer, int x0, int y0)
{
    uint32_t *pixel = (uint32_t *)buffer->memory + (int)y0 * buffer->info.bmiHeader.biWidth + (int)x0;

    *pixel = INT_MAX;
}

/* Just your average matrix multiplication */    
__global__ void kernelTransform(double *transformation, VERTEX *vertices, VERTEX *transformedVertices)
{
    
    VERTEX vertex = vertices[blockIdx.x];


    switch (threadIdx.x)
    {
    case 0:
        transformedVertices[blockIdx.x].x = (transformation[0] * vertex.x) + (transformation[1] * vertex.y) + (transformation[2] * vertex.z) + transformation[3];
        break;
    case 1:
        transformedVertices[blockIdx.x].y = (transformation[4] * vertex.x) + (transformation[5] * vertex.y) + (transformation[6] * vertex.z) + transformation[7];
        break;
    case 2:
        transformedVertices[blockIdx.x].z = (transformation[8] * vertex.x) + (transformation[9] * vertex.y) + (transformation[10] * vertex.z) + transformation[11];
        break;
    }
}

/* Composes two 4x4 transformation matrices */
__global__ void kernelCompose(double *a, double *b, double *result)
{
    int row = threadIdx.y;
    int col = threadIdx.x;
    double c_value = 0;
    for (int i = 0; i < 4; i++)
    {
        c_value += a[row * 4 + i] * b[i * 4 + col];
    }
    result[row * 4 + col] = c_value;
}