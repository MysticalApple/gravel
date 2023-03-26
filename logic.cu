#include "logic.h"
#include "rendering.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
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
    bool dPadUp = (gamepad.wButtons & XINPUT_GAMEPAD_DPAD_UP);
    bool dPadDown = (gamepad.wButtons & XINPUT_GAMEPAD_DPAD_DOWN);
    bool dPadLeft = (gamepad.wButtons & XINPUT_GAMEPAD_DPAD_LEFT);
    bool dPadRight = (gamepad.wButtons & XINPUT_GAMEPAD_DPAD_RIGHT);
    bool start = (gamepad.wButtons & XINPUT_GAMEPAD_START);
    bool back = (gamepad.wButtons & XINPUT_GAMEPAD_BACK);
    bool shoulderLeft = (gamepad.wButtons & XINPUT_GAMEPAD_LEFT_SHOULDER);
    bool shoulderRight = (gamepad.wButtons & XINPUT_GAMEPAD_RIGHT_SHOULDER);
    bool buttonA = (gamepad.wButtons & XINPUT_GAMEPAD_A);
    bool buttonB = (gamepad.wButtons & XINPUT_GAMEPAD_B);
    bool buttonX = (gamepad.wButtons & XINPUT_GAMEPAD_X);
    bool buttonY = (gamepad.wButtons & XINPUT_GAMEPAD_Y);

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

    const int speed = 1;
    const size_t transformationMatrixSize = 4 * 4 * sizeof(double);
    dim3 matrixDim(4, 4);

    static double transformation[4 * 4] = {1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1, 0,
                                           0, 0, 0, 1};

    double inputTransformation[4 * 4] = {1, 0, 0, 0,
                                         0, 1, 0, 0,
                                         0, 0, 1, 0,
                                         0, 0, 0, 1};

    double *devInputTransformation;
    cudaMalloc(&devInputTransformation, transformationMatrixSize);
    cudaMemcpy(devInputTransformation, inputTransformation, transformationMatrixSize, cudaMemcpyHostToDevice);

    double *devResult;
    cudaMalloc(&devResult, transformationMatrixSize);

    /* Panning */
    if (abs(thumbStickRightX) > XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE)
    {
        double theta = copysign(speed, thumbStickRightX) * (abs(thumbStickRightX) - XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE) / (double)(SHRT_MAX - XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE);
        printf("%lf\n", theta);
        theta = theta * 2 * M_PI / 360;

        double rotateY[4 * 4] = {cos(theta), 0, -sin(theta), 0,
                                 0, 1, 0, 0,
                                 sin(theta), 0, cos(theta), 0,
                                 0, 0, 0, 1};

        double *devRotateY;
        cudaMalloc(&devRotateY, transformationMatrixSize);
        cudaMemcpy(devRotateY, rotateY, transformationMatrixSize, cudaMemcpyHostToDevice);

        kernelCompose<<<1, matrixDim>>>(devRotateY, devInputTransformation, devResult);

        cudaMemcpy(devInputTransformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToDevice);

        cudaFree(rotateY);
    }

    /* Tilting */
    if (abs(thumbStickRightY) > XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE)
    {
        double theta = copysign(speed, thumbStickRightY) * (abs(thumbStickRightY) - XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE) / (double)(SHRT_MAX - XINPUT_GAMEPAD_RIGHT_THUMB_DEADZONE);
        printf("%lf\n", theta);
        theta = theta * 2 * M_PI / 360;

        double rotateX[4 * 4] = {1, 0, 0, 0,
                                 0, cos(theta), -sin(theta), 0,
                                 0, sin(theta), cos(theta), 0,
                                 0, 0, 0, 1};

        double *devRotateX;
        cudaMalloc(&devRotateX, transformationMatrixSize);
        cudaMemcpy(devRotateX, rotateX, transformationMatrixSize, cudaMemcpyHostToDevice);

        kernelCompose<<<1, matrixDim>>>(devRotateX, devInputTransformation, devResult);

        cudaMemcpy(devInputTransformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToDevice);

        cudaFree(rotateX);
    }

    /* Moving along x-axis */
    if (abs(thumbStickLeftX) > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE)
    {
        double translateX[4 * 4] = {1, 0, 0, -copysign(speed, thumbStickLeftX) * (abs(thumbStickLeftX) - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE) / (double)(SHRT_MAX - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE),
                                    0, 1, 0, 0,
                                    0, 0, 1, 0,
                                    0, 0, 0, 1};

        double *devTranslateX;
        cudaMalloc(&devTranslateX, transformationMatrixSize);
        cudaMemcpy(devTranslateX, translateX, transformationMatrixSize, cudaMemcpyHostToDevice);

        kernelCompose<<<1, matrixDim>>>(devTranslateX, devInputTransformation, devResult);

        cudaMemcpy(devInputTransformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToDevice);
        
        cudaFree(devTranslateX);
    }

    /* Moving along z-axis */
    if (abs(thumbStickLeftY) > XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE)
    {
        double translateZ[4 * 4] = {1, 0, 0, 0,
                                    0, 1, 0, 0,
                                    0, 0, 1, -copysign(speed, thumbStickLeftY) * (abs(thumbStickLeftY) - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE) / (double)(SHRT_MAX - XINPUT_GAMEPAD_LEFT_THUMB_DEADZONE),
                                    0, 0, 0, 1};

        double *devTranslateZ;
        cudaMalloc(&devTranslateZ, transformationMatrixSize);
        cudaMemcpy(devTranslateZ, translateZ, transformationMatrixSize, cudaMemcpyHostToDevice);

        kernelCompose<<<1, matrixDim>>>(devTranslateZ, devInputTransformation, devResult);

        cudaMemcpy(devInputTransformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToDevice);

        cudaFree(devTranslateZ);
    }

    /* Moving along y-axis */
    if (buttonA || buttonB)
    {
        double translateY[4 * 4] = {1, 0, 0, 0,
                                    0, 1, 0, (double)speed * (buttonA - buttonB),
                                    0, 0, 1, 0,
                                    0, 0, 0, 1};

        double *devTranslateY;
        cudaMalloc(&devTranslateY, transformationMatrixSize);
        cudaMemcpy(devTranslateY, translateY, transformationMatrixSize, cudaMemcpyHostToDevice);

        kernelCompose<<<1, matrixDim>>>(devTranslateY, devInputTransformation, devResult);

        cudaMemcpy(devInputTransformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToDevice);

        cudaFree(devTranslateY);
    }

    /* Resets camera */
    if (back)
    {
        double identity[4 * 4] = {1, 0, 0, 0,
                                  0, 1, 0, 0,
                                  0, 0, 1, 0,
                                  0, 0, 0, 1};
                                  
        memcpy(transformation, identity, transformationMatrixSize);
    }

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

    double *devTransformation;
    cudaMalloc(&devTransformation, transformationMatrixSize);
    cudaMemcpy(devTransformation, transformation, transformationMatrixSize, cudaMemcpyHostToDevice);

    kernelCompose<<<1, matrixDim>>>(devInputTransformation, devTransformation, devResult);

    cudaMemcpy(transformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(devTransformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToDevice);
    
    cudaFree(devInputTransformation);

    VERTEX transformedVertices[vertexCount];

    /* ===================== *
     *    RENDERING ZONE     *
     * ===================== */

    /* Changes origin from bottom left corner to center of screen */
    double screenTransform[4 * 4] = {1, 0, 0, (double) buffer->info.bmiHeader.biWidth / 2,
                                     0, 1, 0, (double) buffer->info.bmiHeader.biHeight / 2,
                                     0, 0, 1, 0,
                                     0, 0, 0, 1};
    
    double *devScreenTransform;
    cudaMalloc(&devScreenTransform, transformationMatrixSize);
    cudaMemcpy(devScreenTransform, screenTransform, transformationMatrixSize, cudaMemcpyHostToDevice);


    kernelCompose<<<1, matrixDim>>>(devScreenTransform, devTransformation, devResult);

    cudaFree(devTransformation);
    cudaFree(devScreenTransform);

    /* Transforms all vertices */
    VERTEX *devVertices;
    cudaMalloc(&devVertices, vertexCount * sizeof(VERTEX));
    cudaMemcpy(devVertices, vertices, vertexCount * sizeof(VERTEX), cudaMemcpyHostToDevice);

    VERTEX *devTransformedVertices;
    cudaMalloc(&devTransformedVertices, vertexCount * sizeof(VERTEX));

    kernelTransform<<<vertexCount, 3>>>(devResult, devVertices, devTransformedVertices);

    cudaMemcpy(transformedVertices, devTransformedVertices, vertexCount * sizeof(VERTEX), cudaMemcpyDeviceToHost);

    cudaFree(devVertices);
    cudaFree(devTransformedVertices);
    cudaFree(devResult);


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

        if (transformedVertices[edge.a].z < 0 || transformedVertices[edge.b].z < 0) continue;

        int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy, e2; /* error value e_xy */

        while (true)
        {
            if (x0 < buffer->info.bmiHeader.biWidth && x0 >= 0 && y0 < buffer->info.bmiHeader.biHeight && y0 >= 0)
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
