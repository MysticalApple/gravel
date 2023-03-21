#include <stdio.h>
#include "rendering.h"

__global__ void kernelHello(void)
{
    printf("Hello, World!\n");
}

void HelloWorld(void)
{
    kernelHello<<<1, 1>>>();
}
