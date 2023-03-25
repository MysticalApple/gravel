#include "logic.h"
#include "rendering.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

static void DrawPoint(win32_offscreen_buffer *buffer, int x0, int y0)
{
    uint32_t *pixel = (uint32_t *)buffer->memory + (int)y0 * buffer->info.bmiHeader.biWidth + (int)x0;

    *pixel = INT_MAX;
}

/* NOTE: The following code is completely illogical. */
void HandleLogic(win32_offscreen_buffer *buffer, XINPUT_GAMEPAD gamepad, time_t timeInit)
{
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
    const static int edgeCount = 12;

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

    static EDGE edges[edgeCount] = {
        {0, 3},
        {0, 4},
        {0, 7},
        {1, 3},
        {1, 5},
        {1, 7},
        {2, 3},
        {2, 4},
        {2, 5},
        {6, 4},
        {6, 5},
        {6, 7}
    };

    static double transformation[4 * 4] = {0.866025, 0, -0.5, 400,
                                           -0.25, 0.866025, -0.433013, 400,
                                           0.433013, 0.5, 0.75, 1000,
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

    kernelTransform<<<vertexCount, 3>>>(devTransformation, devVertices, devTransformedVertices);

    cudaMemcpy(transformedVertices, devTransformedVertices, vertexCount * sizeof(VERTEX), cudaMemcpyDeviceToHost);

    cudaFree(devVertices);
    cudaFree(devTransformedVertices);
    cudaFree(devTransformation);


    /* Clears window to black */
    memset(buffer->memory, 0, buffer->info.bmiHeader.biWidth * buffer->info.bmiHeader.biHeight * buffer->bytesPerPixel);

    /* Bresenham Line Drawing Algorithm */
    /* Copied from https://gist.github.com/bert/1085538#file-plot_line-c */
    for (int i = 0; i < edgeCount; i++)
    {
        EDGE edge = edges[i];

        
        int x0 = (int)transformedVertices[edge.a].x;
        int y0 = (int)transformedVertices[edge.a].y;
        int x1 = (int)transformedVertices[edge.b].x;
        int y1 = (int)transformedVertices[edge.b].y;

        int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2; /* error value e_xy */

        while (true)
        {
            DrawPoint(buffer, x0, y0);

            if (x0 == x1 && y0 == y1) break;
            
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

}
