#include "logic.h"
#include "rendering.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

// static int8_t sign(int32_t x)
// {
//     return x >> 31 ? -1 : 1;
// }

/* NOTE: The following code is completely illogical. */
void HandleLogic(win32_offscreen_buffer *buffer, XINPUT_GAMEPAD gamepad, time_t timeInit)
{
    static double xOffset;
    static double yOffset;

    /* ========== *
     *  FPS INFO  *
     * ========== */
    time_t secondsSinceStart = time(NULL) - timeInit;

    static unsigned int frames;
    frames++;

    static bool printedFPS;
    if (secondsSinceStart % 5)
        printedFPS = false;

    if (secondsSinceStart % 5 == 0 && !printedFPS)
    {
        printf("%f fps\n", frames / 5.0);
        frames = 0;
        printedFPS = true;
    }

    /* ================ *
     *  INPUT HANDLING  *
     * ================ */
    BOOL dPadUp = (gamepad.wButtons & XINPUT_GAMEPAD_DPAD_UP);
    BOOL dPadDown = (gamepad.wButtons & XINPUT_GAMEPAD_DPAD_DOWN);
    BOOL dPadLeft = (gamepad.wButtons & XINPUT_GAMEPAD_DPAD_LEFT);
    BOOL dPadRight = (gamepad.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT);
    BOOL start = (gamepad.wButtons & XINPUT_GAMEPAD_START);
    BOOL back = (gamepad.wButtons & XINPUT_GAMEPAD_BACK);
    BOOL shoulderLeft = (gamepad.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER);
    BOOL shoulderRight = (gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER);
    BOOL buttonA = (gamepad.wButtons & XINPUT_GAMEPAD_A);
    BOOL buttonB = (gamepad.wButtons & XINPUT_GAMEPAD_B);
    BOOL buttonX = (gamepad.wButtons & XINPUT_GAMEPAD_X);
    BOOL buttonY = (gamepad.wButtons & XINPUT_GAMEPAD_Y);

    int16_t thumbStickLeftX = gamepad.sThumbLX;
    int16_t thumbStickLeftY = gamepad.sThumbLY;
    int16_t thumbStickRightX = gamepad.sThumbRX;
    int16_t thumbStickRightY = gamepad.sThumbRY;

    uint8_t triggerLeft = gamepad.bLeftTrigger;
    uint8_t triggerRight = gamepad.bRightTrigger;

    //    XINPUT_VIBRATION Vibration;
    //    Vibration.wLeftMotorSpeed = triggerLeft * USHRT_MAX / UCHAR_MAX;
    //    Vibration.wRightMotorSpeed = triggerRight * USHRT_MAX / UCHAR_MAX;
    //    XInputSetState(0, &Vibration);

    // const int speed = 5;

    // if (abs(thumbStickLeftX) > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE)
    // {
    //     xOffset +=
    //         speed * sign(thumbStickLeftX) * (abs(thumbStickLeftX) - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE) /
    //         (double)(SHRT_MAX - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE);
    // }

    // if (abs(thumbStickLeftY) > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE)
    // {
    //     yOffset +=
    //         speed * sign(thumbStickLeftY) * (abs(thumbStickLeftY) - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE) /
    //         (double)(SHRT_MAX - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE);
    // }

    /* Change to reading from file later
       Probably move to main, only needs to run once */
    const static int vertexCount = 8;

    static VERTEX vertices[vertexCount] =
        {
            {400, 0, 0},
            {0, 400, 0},
            {0, 0, 400},
            {0, 0, 0},
            {400, 0, 400},
            {0, 400, 400},
            {400, 400, 400},
            {400, 400, 0},
        };

    static double transformation[4 * 4] = {1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1, 0,
                                           0, 0, 0, 1};

    VERTEX transformedVertices[vertexCount];


    /* ===================== *
     *    RENDERING ZONE     *
     * ===================== */

    /* Allocates all necessary GPU memory*/
    VERTEX *devVertices;
    cudaMalloc(&devVertices, vertexCount * sizeof(VERTEX));
    cudaMemcpy(devVertices, vertices, vertexCount * sizeof(VERTEX), cudaMemcpyHostToDevice);

    double *devTransformation;
    cudaMalloc(&devTransformation, 4 * 4 * sizeof(double));
    cudaMemcpy(devTransformation, transformation, 4 * 4 * sizeof(double), cudaMemcpyHostToDevice);

    VERTEX *devTransformedVertices;
    cudaMalloc(&devTransformedVertices, vertexCount * sizeof(VERTEX));

    kernelTransform<<<1, vertexCount>>>(devTransformation, devVertices, devTransformedVertices);

    cudaMemcpy(transformedVertices, devTransformedVertices, vertexCount * sizeof(VERTEX), cudaMemcpyDeviceToHost);

    /* Clears window to black */
    memset(buffer->memory, 0, buffer->info.bmiHeader.biWidth * buffer->info.bmiHeader.biHeight * buffer->bytesPerPixel);

    /* Draws each point to the window as a single white pixel
       Might change to be a slightly larger circle later      */
    for (int i = 0; i < vertexCount; i++)
    {
        VERTEX vertex = transformedVertices[i];
        //printf("X: %f, Y: %f, Z: %f\n", vertex.x, vertex.y, vertex.z);

        if(vertex.x < 0 || vertex.y < 0 || vertex.z < 0) continue;

        uint32_t *pixel = (uint32_t *)buffer->memory + (int)vertex.y * buffer->info.bmiHeader.biWidth + (int)vertex.x;

        *pixel = INT_MAX;
    }


    // int bitmapMemorySize = buffer->info.bmiHeader.biWidth * buffer->info.bmiHeader.biHeight * buffer->bytesPerPixel;

    // void *devBufferMemory;
    // cudaMalloc(&devBufferMemory, bitmapMemorySize);
    // cudaMemcpy(devBufferMemory, buffer->memory, bitmapMemorySize, cudaMemcpyHostToDevice);

    // kernelDrawPixels<<<buffer->info.bmiHeader.biHeight, buffer->info.bmiHeader.biWidth>>>(devBufferMemory,
    //                                                                                       buffer->info.bmiHeader.biWidth, buffer->info.bmiHeader.biHeight,
    //                                  /* Wide code is good, right? */                      xOffset, yOffset);

    // cudaMemcpy(buffer->memory, devBufferMemory, bitmapMemorySize, cudaMemcpyDeviceToHost);
    // cudaFree(devBufferMemory);

    /* Keeping non-parallel version for my own sanity */

    // uint8_t *row = (uint8_t *)buffer->memory;

    // for (int y = 0; y < buffer->info.bmiHeader.biHeight; y++)
    // {
    //     uint32_t *pixel = (uint32_t *)row;
    //     for (int x = 0; x < buffer->info.bmiHeader.biWidth; x++)
    //     {
    //         uint8_t red, green, blue;

    //         red = x - (int8_t)xOffset;
    //         green = y - (int8_t)yOffset;
    //         blue = (int8_t)xOffset + (y * (int8_t)yOffset);

    //         if (!(uint8_t)(x + (uint8_t)xOffset) || !(uint8_t)(y + (uint8_t)yOffset))
    //         {
    //             red = 255;
    //             green = 255;
    //             blue = 255;
    //         }

    //         *pixel++ = (red << 16) | (green << 8) | (blue << 0); // pixel format is xRGB
    //     }

    //     row += buffer->info.bmiHeader.biWidth * buffer->bytesPerPixel;
    // }
}
