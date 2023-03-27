#pragma once
#include <stdio.h>

typedef struct
{
    double x;
    double y;
    double z;
} VERTEX;

typedef struct
{
    unsigned short a;
    unsigned short b;
} EDGE;

void parseVertices(FILE *object, VERTEX *vertices, const unsigned int vertexCount);
void parseEdges(FILE *object, EDGE *edges, const unsigned int edgeCount);