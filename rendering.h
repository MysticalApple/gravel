#pragma once
#include <windows.h>

typedef struct
{
    BITMAPINFO info;
    void *memory;
    int bytesPerPixel;
} win32_offscreen_buffer;

__global__ void kernelDrawPixels(void *memory, int width, int height, double xOffset, double yOffset);
