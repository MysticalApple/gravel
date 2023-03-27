#pragma once
#include "parsing.h"
#include <windows.h>

typedef struct
{
    BITMAPINFO info;
    void *memory;
    int bytesPerPixel;
} win32_offscreen_buffer;

__global__ void kernelTransform(double *transformation, VERTEX *vertices, VERTEX *transformedVertices);
__global__ void kernelCompose(double *a, double *b, double *result);