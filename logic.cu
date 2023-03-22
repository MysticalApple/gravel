#include "logic.h"
#include "rendering.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

static int8_t sign(int32_t x)
{
    return x >> 31 ? -1 : 1;
}

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

    const int speed = 5;

    if (abs(thumbStickLeftX) > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE)
    {
        xOffset +=
            speed * sign(thumbStickLeftX) * (abs(thumbStickLeftX) - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE) /
            (double)(SHRT_MAX - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE);
    }

    if (abs(thumbStickLeftY) > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE)
    {
        yOffset +=
            speed * sign(thumbStickLeftY) * (abs(thumbStickLeftY) - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE) /
            (double)(SHRT_MAX - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE);
    }

    /* ===================== *
     *    RENDERING ZONE     *
     * ===================== */

    int bitmapMemorySize = buffer->info.bmiHeader.biWidth * buffer->info.bmiHeader.biHeight * buffer->bytesPerPixel;

    void *devBufferMemory;
    cudaMalloc(&devBufferMemory, bitmapMemorySize);
    cudaMemcpy(devBufferMemory, buffer->memory, bitmapMemorySize, cudaMemcpyHostToDevice);

    kernelDrawPixels<<<buffer->info.bmiHeader.biHeight, buffer->info.bmiHeader.biWidth>>>(devBufferMemory,
                                                                                          buffer->info.bmiHeader.biWidth, buffer->info.bmiHeader.biHeight,
                                     /* Wide code is good, right? */                      xOffset, yOffset);

    cudaMemcpy(buffer->memory, devBufferMemory, bitmapMemorySize, cudaMemcpyDeviceToHost);
    cudaFree(devBufferMemory);

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

