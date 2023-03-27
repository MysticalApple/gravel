#include "rendering.h"
#include <stdint.h>

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

/* Bresenham Line Drawing Algorithm
   Copied from https://gist.github.com/bert/1085538#file-plot_line-c */
__global__ void kernelDrawLine(void *bufferMemory, int width, int height, VERTEX *transformedVertices, EDGE *edges)
{
    EDGE edge = edges[blockIdx.x];

    int x0 = (int)transformedVertices[edge.a].x;
    int y0 = (int)transformedVertices[edge.a].y;
    int x1 = (int)transformedVertices[edge.b].x;
    int y1 = (int)transformedVertices[edge.b].y;

    if (transformedVertices[edge.a].z < 0 || transformedVertices[edge.b].z < 0)
        return;

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2; /* error value e_xy */

    while (true)
    {
        if (x0 < width && x0 >= 0 && y0 < height && y0 >= 0)
        {
            uint32_t *pixel = (uint32_t *)bufferMemory + (int)y0 * width + (int)x0;

            *pixel = INT_MAX;
        }

        if (x0 == x1 && y0 == y1)
            break;

        e2 = 2 * err;
        if (e2 >= dy)
        {
            err += dy;
            x0 += sx;
        } /* e_xy+e_x > 0 */

        if (e2 <= dx)
        {
            err += dx;
            y0 += sy;
        } /* e_xy+e_y < 0 */
    }
}