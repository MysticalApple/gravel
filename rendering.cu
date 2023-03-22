#include <stdint.h>

#include "rendering.h"

__global__ void kernelDrawPixels(void *memory, int width, int height, double xOffset, double yOffset)
{
    int x = threadIdx.x;
    int y = blockIdx.x;

    uint8_t red, green, blue;

    red = x - (int8_t)xOffset;
    green = y - (int8_t)yOffset;
    blue = (int8_t)xOffset + (y * (int8_t)yOffset);

    if (!((x + (int8_t)xOffset) % 256) || !((y + (int8_t)yOffset) % 256))
    {
        red = 255;
        green = 255;
        blue = 255;
    }

    uint32_t *pixel = (uint32_t *)memory;
    pixel += y * width + x;

    *pixel = (red << 16) | (green << 8) | (blue << 0);
}
