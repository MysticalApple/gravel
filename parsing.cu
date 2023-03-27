#include "parsing.h"

/* Populate the vertices array with coords from the file */
int parseVertices(FILE *object, VERTEX *vertices, const unsigned int vertexCount) {
    VERTEX *vertices = (vertex *)malloc(vertexCount * sizeof(vertex));
    int c, prev_c = EOF;
    for (int i = 0; i < count; i++) {
        if (fscanf(points, "%lf %lf %lf", &vertices[i].x, &vertices[i].y, &vertices[i].z) != 3) {
            return EOF;
        }
        c = getc(points);
        if(c == '\n' && prev_c == '\n') {
            return c;
        }
        prev_c = c;
    }
    return EOF;
}

/* Populate the edges array with edges from the file */
int parseEdges(FILE *object, EDGE *edges, const unsigned int edgeCount, int count) {
    EDGE *edges = (edge *)malloc(edgeCount * sizeof(edge));
     int c, prev_c = EOF;
    for (int i = 0; i < count; i++) {
        if (fscanf(points, "%hu %hu", &wireframes[i].a, &wireframes[i].b) != 2) {
            return EOF;
        }
        c = getc(points);
        if(c == '\n' && prev_c == '\n') {
            return c;
        }
        prev_c = c;
    }
    return EOF;
}
