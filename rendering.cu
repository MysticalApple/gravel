#include "rendering.h"

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

