#include "logic.h"
#include "rendering.h"

#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <time.h>

/* NOTE: The following code is completely illogical. */
void HandleLogic(win32_offscreen_buffer *buffer, XINPUT_GAMEPAD gamepad, VERTEX *vertices, EDGE *edges, const unsigned int vertexCount, const unsigned int edgeCount, time_t timeInit)
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

    /* Dilating */
    if (buttonX ^ buttonY)
    {
        double dilationFactor = buttonX ? 1 + (speed * 0.005) : 1 - (speed * 0.005);

        double dilate[4 * 4] = {dilationFactor, 0, 0, 0,
                                0, dilationFactor, 0, 0,
                                0, 0, dilationFactor, 0,
                                0, 0, 0, 1};

        double *devDilate;
        cudaMalloc(&devDilate, transformationMatrixSize);
        cudaMemcpy(devDilate, dilate, transformationMatrixSize, cudaMemcpyHostToDevice);

        kernelCompose<<<1, matrixDim>>>(devDilate, devInputTransformation, devResult);

        cudaMemcpy(devInputTransformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToDevice);

        cudaFree(dilate);
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

    double *devTransformation;
    cudaMalloc(&devTransformation, transformationMatrixSize);
    cudaMemcpy(devTransformation, transformation, transformationMatrixSize, cudaMemcpyHostToDevice);

    kernelCompose<<<1, matrixDim>>>(devInputTransformation, devTransformation, devResult);

    cudaMemcpy(transformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(devTransformation, devResult, transformationMatrixSize, cudaMemcpyDeviceToDevice);
    
    cudaFree(devInputTransformation);

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

    cudaFree(devVertices);
    cudaFree(devResult);


    /* Clears window to black */
    void *devBufferMemory;
    cudaMalloc(&devBufferMemory, (buffer->info.bmiHeader.biWidth * buffer->info.bmiHeader.biHeight) * buffer->bytesPerPixel);

    EDGE *devEdges;
    cudaMalloc(&devEdges, edgeCount * sizeof(EDGE));
    cudaMemcpy(devEdges, edges, edgeCount * sizeof(EDGE), cudaMemcpyHostToDevice);

    kernelDrawLine<<<edgeCount, 1>>>(devBufferMemory, buffer->info.bmiHeader.biWidth, buffer->info.bmiHeader.biHeight, devTransformedVertices, devEdges);

    cudaMemcpy(buffer->memory, devBufferMemory, (buffer->info.bmiHeader.biWidth * buffer->info.bmiHeader.biHeight) * buffer->bytesPerPixel, cudaMemcpyDeviceToHost);

    cudaFree(devBufferMemory);
    cudaFree(devTransformedVertices);
    cudaFree(devEdges);
}
