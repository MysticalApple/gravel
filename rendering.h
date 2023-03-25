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

typedef struct
{
    unsigned short a;
    unsigned short b;
} EDGE;


__global__ void kernelTransform(double *transformation, VERTEX *vertices, VERTEX *transformedVertices);
