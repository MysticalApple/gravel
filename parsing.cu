#include "parsing.h"

/* Populate the vertices array with coords from the file */
void parseVertices(FILE *object, VERTEX *vertices, const unsigned int vertexCount) {
    
    for (int i = 0; i < vertexCount; i++)
    {
        fscanf(object, "%lf %lf %lf\n", &vertices[i].x, &vertices[i].y, &vertices[i].z);
    }

}

/* Populate the edges array with edges from the file */
void parseEdges(FILE *object, EDGE *edges, const unsigned int edgeCount) {
    for (int i = 0; i < edgeCount; i++)
    {
        fscanf(object, "%hu %hu", &edges[i].a, &edges[i].b);
    }
}
