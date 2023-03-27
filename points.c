#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double x;
    double y;
    double z;
} vertex;

typedef struct {
    unsigned short punto1;
    unsigned short punto2;
} edge;

int parse1(FILE *points, unsigned short *nopoints, int count) {
    int c, prev_c = EOF;
    for (int i = 0; i < count; i++) {
        if (fscanf(points, "%hu", &nopoints[i]) != 1) {
            return EOF;
        }
        c = getc(points);
        if (c == '\n' && prev_c == '\n') {
            return c;
        }
        prev_c = c;
    }
    return EOF;
}

int parse2(FILE *points, vertex *vertices, int count) {
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

int parse3(FILE *points, edge *wireframes, int count) {
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

int parse4(FILE *points, unsigned short *noedge, int count) {
    int c, prev_c = EOF;
    for (int i = 0; i < count-1; i++) {
        if (fscanf(points, "%hu", &noedge[i]) != 1) {
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

int main(void) {
    FILE *points;
    int count;
    char fname[128];
    printf("Which object file do you want to read?\n");
    scanf("%127s", fname);
    strcat(fname,".txt");

    points = fopen(fname, "r");
    if (points == NULL) {
        perror(fname);
        exit(1);
    }
    fscanf(points, "%d", &count);
    unsigned short *nopoints = (unsigned short *)malloc(count * sizeof(unsigned short));
    vertex *vertices = (vertex *)malloc(count * sizeof(vertex));
    edge *wireframes = (edge *)malloc(count * sizeof(edge));
    unsigned short *noedgeval = (unsigned short *)malloc((count-1) * sizeof(unsigned short));
    parse1(points, nopoints, count);
    parse2(points, vertices, count);
    parse3(points, wireframes, count);
    parse4(points, noedgeval, count);
    fclose(points);

    for (int i = 0; i < count; i++) {
        printf("%hu\n", nopoints[i]);
        vertex point = vertices[i];
        printf("%lf %lf %lf\n", point.x, point.y, point.z);
        edge line = wireframes[i];
        printf("%hu %hu\n", line.punto1, line.punto2);
        printf("%hu\n", noedgeval[i]);
    }

    free(nopoints);
    free(vertices);
    free(wireframes);
    free(noedgeval);
   
    return 0;
}