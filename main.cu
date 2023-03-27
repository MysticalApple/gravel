/*
    A large part of the window and input code is 
    stolen from MollyRocket's HandmadeHero series.
*/

#ifndef UNICODE
#define UNICODE
#endif

#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <windows.h>
#include <xinput.h>
#include <stdbool.h>

#include "rendering.h"
#include "logic.h"
#include "parsing.h"

#define internal static

/* idk wtf this x input stuff is but it works */
#define X_INPUT_GET_STATE(name) int64_t WINAPI name(DWORD dwUserIndex, XINPUT_STATE *pState)

typedef X_INPUT_GET_STATE(x_input_get_state);

X_INPUT_GET_STATE(XInputGetStateStub)
{
    (void)dwUserIndex;
    (void)pState;
    return ERROR_DEVICE_NOT_CONNECTED;
}

static x_input_get_state *XInputGetState_ = XInputGetStateStub;
#define XInputGetState XInputGetState_

#define X_INPUT_SET_STATE(name) int64_t WINAPI name(DWORD dwUserIndex, XINPUT_VIBRATION *pVibration)

typedef X_INPUT_SET_STATE(x_input_set_state);

X_INPUT_SET_STATE(XInputSetStateStub)
{
    (void)dwUserIndex;
    (void)pVibration;
    return ERROR_DEVICE_NOT_CONNECTED;
}

static x_input_set_state *XInputSetState_ = XInputSetStateStub;
#define XInputSetState XInputSetState_

static void win32_LoadXInput(void)
{
    HMODULE XInputLibrary = LoadLibraryA("xinput1_4.dll");
    if (XInputLibrary)
    {
        XInputGetState = (x_input_get_state *)GetProcAddress(XInputLibrary, "XInputGetState");
        if (!XInputGetState)
            XInputGetState = XInputGetStateStub;

        XInputSetState = (x_input_set_state *)GetProcAddress(XInputLibrary, "XInputSetState");
        if (!XInputSetState)
            XInputSetState = XInputSetStateStub;
    }
}

static win32_offscreen_buffer globalBackBuffer;

LRESULT CALLBACK win32_WindowProc(HWND windowHandle, UINT uMsg, WPARAM wParam, LPARAM lParam);

internal void win32_ResizeDIBSection(win32_offscreen_buffer *buffer, int width, int height);
internal void win32_CopyBufferToWindow(HDC deviceContext, win32_offscreen_buffer *buffer, int windowWidth, int windowHeight);

typedef struct
{
    int width;
    int height;
} win32_window_dimensions;
win32_window_dimensions win32_GetWindowDimensions(HWND windowHandle);

#define WIDTH 1920
#define HEIGHT 1080

static BOOL running;

/* Main Windows entrypoint */
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PSTR lpCmdLine, INT nCmdShow)
{
    /* Unused WinMain variables to avoid compiler complaints */
    (void)hPrevInstance;

    const time_t timeInit = time(NULL);

    FILE *object = fopen(lpCmdLine, "r");
    if (!object)
    {
        printf("No such file exists.");
        return 1;
    }

    /* File parsing into vertices and edges */
    unsigned int vertexCount;
    unsigned int edgeCount;

    fscanf(object, "%u\n", &vertexCount);
    VERTEX *vertices = (VERTEX *)malloc(vertexCount * sizeof(VERTEX));
    parseVertices(object, vertices, vertexCount);

    fscanf(object, "%u\n", &edgeCount);
    EDGE *edges = (EDGE *)malloc(edgeCount * sizeof(EDGE));
    parseEdges(object, edges, edgeCount);

    fclose(object);

    /* hehe funny xbox controller*/
    win32_LoadXInput();

    /* Registers the window class */
    const wchar_t CLASS_NAME[] = L"Main Window Class";

    WNDCLASS wc = {0};

    win32_ResizeDIBSection(&globalBackBuffer, WIDTH, HEIGHT);

    wc.lpfnWndProc = win32_WindowProc;
    wc.hInstance = hInstance;
    wc.lpszClassName = CLASS_NAME;
    wc.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;

    RegisterClass(&wc);

    /* Creates the window */
    HWND windowHandle = CreateWindowEx(
        0,
        CLASS_NAME,
        L"Graphics Testing",
        WS_OVERLAPPEDWINDOW ^ WS_THICKFRAME ^ WS_MAXIMIZEBOX,

        /* Size and position */
        CW_USEDEFAULT, CW_USEDEFAULT, WIDTH + 16, HEIGHT + 39, // TODO: Portability

        NULL,
        NULL,
        hInstance,
        NULL);

    if (!windowHandle)
    {
        return 0;
    }

    ShowWindow(windowHandle, nCmdShow);

    HDC deviceContext = GetDC(windowHandle);

    /* Runs the message loop */
    running = true;

    while (running)
    {
        // time_t start = clock();

        MSG msg;
        while (PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
        {
            if (msg.message == WM_QUIT)
            {
                running = false;
            }

            TranslateMessage(&msg);
            DispatchMessage(&msg);
        }

        XINPUT_GAMEPAD pad;
        for (DWORD controllerIndex = 0; controllerIndex < XUSER_MAX_COUNT; controllerIndex++)
        {
            XINPUT_STATE controllerState;

            if (XInputGetState(controllerIndex, &controllerState) == ERROR_SUCCESS)
            {
                /* Controller is plugged in */
                pad = controllerState.Gamepad;
            }
            else
            {
                /* No controller is plugged in */
            }
        }

        HandleLogic(&globalBackBuffer, pad, vertices, edges, vertexCount, edgeCount, timeInit);

        win32_window_dimensions dimensions = win32_GetWindowDimensions(windowHandle);
        win32_CopyBufferToWindow(deviceContext, &globalBackBuffer,
                                 dimensions.width, dimensions.height);

        // time_t end = clock();
        // printf("%f milliseconds\n", (double) (1000 * (end - start)) / CLOCKS_PER_SEC);
    }

    ReleaseDC(windowHandle, deviceContext);

    return 0;
}

/* Message loop */
LRESULT CALLBACK win32_WindowProc(HWND windowHandle, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg)
    {
    /* When the red X is pressed (or alt+f4, or close from menu) */
    case WM_CLOSE:
    {
        running = false;
    }
    break;

    /* When the window is switched to or away from */
    case WM_ACTIVATEAPP:
    {
    }
    break;

    /* When the program closes due to an error? */
    case WM_DESTROY:
    {
        running = false;
    }
    break;

    case WM_SYSKEYDOWN:
    case WM_SYSKEYUP:
    case WM_KEYDOWN:
        break;
    case WM_KEYUP:
    {
        //            uint32_t VKCode = wParam;
        //            BOOL wasDown = lParam & (1 << 30) ? true : false;
        //            BOOL isDown = lParam & (1 << 31) ? true : false;
        //            BOOL altKeyDown = lParam & (1 << 29) ? true : false;
        //
        //            if (VKCode == 'W') {}
        //            if (VKCode == 'S') {}
        //            if (VKCode == 'A') {}
        //            if (VKCode == 'D') {}
        //            if (VKCode == 'Q') {}
        //            if (VKCode == 'E') {}
        //            if (VKCode == VK_UP) {}
        //            if (VKCode == VK_DOWN) {}
        //            if (VKCode == VK_LEFT) {}
        //            if (VKCode == VK_RIGHT) {}
        //            if (VKCode == VK_ESCAPE) {}
        //            if (VKCode == VK_SPACE) {}
        //            if (VKCode == VK_F4 && altKeyDown) running = false;
    }
    break;

    /* When the window is resized */
    case WM_PAINT:
    {
        PAINTSTRUCT paint;
        HDC deviceContext = BeginPaint(windowHandle, &paint);

        win32_window_dimensions dimensions = win32_GetWindowDimensions(windowHandle);
        win32_CopyBufferToWindow(deviceContext, &globalBackBuffer,
                                 dimensions.width, dimensions.height);

        EndPaint(windowHandle, &paint);
    }
    break;

    default:
        break;
    }

    return DefWindowProc(windowHandle, uMsg, wParam, lParam);
}

/* Called when the window is resized */
internal void win32_ResizeDIBSection(win32_offscreen_buffer *buffer, int width, int height)
{
    /* Clears the pre-existing bitmap */
    if (buffer->memory)
    {
        VirtualFree(buffer->memory, 0, MEM_RELEASE);
    }

    buffer->bytesPerPixel = 4; /* RGB + padding for dword alignment */

    /* Creates bitmap */
    buffer->info.bmiHeader.biSize = sizeof(buffer->info.bmiHeader);
    buffer->info.bmiHeader.biWidth = width;
    buffer->info.bmiHeader.biHeight = height;
    buffer->info.bmiHeader.biPlanes = 1;
    buffer->info.bmiHeader.biBitCount = 8 * buffer->bytesPerPixel; /* 8 bits per bytes */
    buffer->info.bmiHeader.biCompression = BI_RGB;

    int bitmapMemorySize = (width * height) * buffer->bytesPerPixel;
    buffer->memory = VirtualAlloc(NULL, bitmapMemorySize,
                                  MEM_COMMIT, PAGE_READWRITE);
}

/* Renders the given buffer onto the given device context thing */
internal void
win32_CopyBufferToWindow(HDC deviceContext, win32_offscreen_buffer *buffer, int windowWidth, int windowHeight)
{

    /* Copies the bitmap to the screen (deviceContext) */
    StretchDIBits(
        deviceContext,
        0, 0, windowWidth, windowHeight,
        0, 0, buffer->info.bmiHeader.biWidth, buffer->info.bmiHeader.biHeight,
        buffer->memory, &buffer->info,
        DIB_RGB_COLORS, SRCCOPY);
}

/* Helper function to get a window's dimensions from its handle */
win32_window_dimensions win32_GetWindowDimensions(HWND windowHandle)
{
    win32_window_dimensions result;

    RECT clientRect;
    GetClientRect(windowHandle, &clientRect);

    result.width = clientRect.right - clientRect.left;
    result.height = clientRect.bottom - clientRect.top;

    //  printf("Resolution: %i x %i\n", result.width, result.height);

    return result;
}
