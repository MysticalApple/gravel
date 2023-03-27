#include "parsing.h"

/* Populate the vertices array with coords from the file */
int parseVertices(FILE *object, VERTEX *vertices, const unsigned int vertexCount, int count) {
    char fname[128];
    printf("Which object file do you want to read?\n");
    scanf("%127s", fname);
    strcat(fname,".txt");
    points = fopen(fname, "r");
    vertex *vertices = (vertex *)malloc(count * sizeof(vertex));
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
    fclose(points);
    free(vertices);
}

/* Populate the edges array with edges from the file */
int parseEdges(FILE *object, EDGE *edges, const unsigned int edgeCount, int count) {
    char fname[128];
    printf("Which object file do you want to read?\n");
    scanf("%127s", fname);
    strcat(fname,".txt");
    points = fopen(fname, "r");
    edge *wireframes = (edge *)malloc(count * sizeof(edge));
     int c, prev_c = EOF;
    for (int i = 0; i < count; i++) {
        if (fscanf(points, "%hu %hu", &wireframes[i].punto1, &wireframes[i].punto2) != 2) {
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
