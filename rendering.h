#pragma once
#include <windows.h>

typedef struct
{
    BITMAPINFO info;
    void *memory;
    int bytesPerPixel;
} win32_offscreen_buffer;

typedef struct
{
    double x;
    double y;
    double z;
} VERTEX;

// __global__ void kernelDrawPixels(void *memory, int width, int height, double xOffset, double yOffset);
__global__ void kernelTransform(double *transformation, VERTEX *vertices, VERTEX *transformedVertices);
