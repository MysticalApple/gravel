#include <stdint.h>

#include <windows.h>
#include <xinput.h>

#include "rendering.h"

void HandleLogic(win32_offscreen_buffer *buffer, XINPUT_GAMEPAD gamepad, VERTEX *vertices, EDGE *edges, const unsigned int vertexCount, const unsigned int edgeCount, time_t timeInit);
