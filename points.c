#include <stdio.h>
#include <stdlib.h>

typedef struct {
    unsigned short g;
} pointno;

typedef struct {
    double x;
    double y;
    double z;
} vertex;

typedef struct {
    unsigned short punto1;
    unsigned short punto2;
} edge;

typedef struct {
    unsigned short p;
} edgeno;

int parse1(FILE *points, pointno *nopoints, int count) {
    int c, prev_c = EOF;
    for (int i = 0; i < count; i++) {
        if (fscanf(points, "%hu", &nopoints[i].g) != 1) {
            return EOF;
        }
        c = getchar();
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
        c = getchar();
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
        fscanf(points, "%hu %hu", &wireframes[i].punto1, &wireframes[i].punto2);
        c = getchar();
        if(c == '\n' && prev_c == '\n') {
            return c;
        }
        prev_c = c;
    }
    return EOF;
}
int parse4(FILE *points, edgeno *noedge, int count) {
    int c, prev_c = EOF;
    for (int i = 0; i < count-1; i++) {
        fscanf(points, "%hu", &noedge[i].p);
        c = getchar();
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

    points = fopen("cube.txt", "r");
    if (points == NULL) {
        perror("cube.txt");
        exit(1);
    }
    fscanf(points, "%d", &count);
    pointno *nopoints = (pointno *)malloc(count *sizeof(pointno));
    parse1(nopoints, nopoints, count);
    vertex *vertices = (vertex *)malloc(count * sizeof(vertex));
    parse2(points, vertices, count);
    edge *wireframes = (edge *)malloc(count *sizeof(edge));
    parse3(points, wireframes, count);
    edgeno *noedge = (edgeno *)malloc(count *sizeof(edgeno));
    parse4(points, noedge, count);
    fclose(points);

    for (int i = 0; i < count; i++) {
        pointno n = nopoints[i];
        printf("%hu\n", n.g);
        vertex point = vertices[i];
        printf("%lf %lf %lf\n", point.x, point.y, point.z);
        edge line = wireframes[i];
        printf("%hu %hu\n", line.punto1, line.punto2);
        edgeno noedgeval = noedge[i];
        printf("%hu\n", noedgeval.p);
    }

    free(nopoints);
    free(vertices);
    free(wireframes);
    free(noedge);

    points = fopen("Pentpyr.txt", "r");
    if (points == NULL) {
        perror("Pentpyr.txt");
        exit(1);
    }

    fscanf(points, "%d", &count);
    nopoints = (pointno *)malloc(count *sizeof(pointno));
    parse1(points, nopoints, count);
    vertices = (vertex *)malloc(count * sizeof(vertex));
    parse2(points, vertices, count);
    wireframes = (edge *)malloc(count *sizeof(edge));
    parse3(points, wireframes, count);
    noedge = (edgeno *)malloc(count *sizeof(edgeno));
    parse4(points, noedge, count);
    fclose(points);

    for (int i = 0; i < count; i++) {
        pointno n = nopoints[i];
        printf("%hu\n", n.g);
        vertex point = vertices[i];
        printf("%lf %lf %lf\n", point.x, point.y, point.z);
        edge line = wireframes[i];
        printf("%hu %hu\n", line.punto1, line.punto2);
        edgeno noedgeval = noedge[i];
        printf("%hu\n", noedgeval.p);
    }

    free(nopoints);
    free(vertices);
    free(wireframes);
    free(noedge);

    points = fopen("tetrahedron.txt", "r");
    if (points == NULL) {
        perror("tetrahedron.txt");
        exit(1);
    }

    fscanf(points, "%d", &count);
    nopoints = (pointno *)malloc(count *sizeof(pointno));
    parse1(points, nopoints, count);
    vertices = (vertex *)malloc(count * sizeof(vertex));
    parse2(points, vertices, count);
    wireframes = (edge *)malloc(count *sizeof(edge));
    parse3(points, wireframes, count);
    noedge = (edgeno *)malloc(count *sizeof(edgeno));
    parse4(points, noedge, count);
    fclose(points);

    for (int i = 0; i < count; i++) {
        pointno n = nopoints[i];
        printf("%hu\n", n.g);
        vertex point = vertices[i];
        printf("%lf %lf %lf\n", point.x, point.y, point.z);
        edge line = wireframes[i];
        printf("%hu %hu\n", line.punto1, line.punto2);
        edgeno noedgeval = noedge[i];
        printf("%hu\n", noedgeval.p);
    }

    free(nopoints);
    free(vertices);
    free(wireframes);
    free(noedge);
   


    points = fopen("triprism.txt", "r");
    if (points == NULL) {
        perror("triprism.txt");
        exit(1);
    }

    fscanf(points, "%d", &count);
    nopoints = (pointno *)malloc(count *sizeof(pointno));
    parse1(points, nopoints, count);
    vertices = (vertex *)malloc(count * sizeof(vertex));
    parse2(points, vertices, count);
    wireframes = (edge *)malloc(count *sizeof(edge));
    parse3(points, wireframes, count);
    noedge = (edgeno *)malloc(count *sizeof(edgeno));
    parse4(points, noedge, count);
    fclose(points);

    for (int i = 0; i < count; i++) {
        pointno n = nopoints[i];
        printf("%hu\n", n.g);
        vertex point = vertices[i];
        printf("%lf %lf %lf\n", point.x, point.y, point.z);
        edge line = wireframes[i];
        printf("%hu %hu\n", line.punto1, line.punto2);
        edgeno noedgeval = noedge[i];
        printf("%hu\n", noedgeval.p);
    }

    free(nopoints);
    free(vertices);
    free(wireframes);
    free(noedge);
   
    return 0;
}

/*int main(void) {
    FILE *points = fopen("sphere.txt", "r");
    if (points == NULL) {
        perror("sphere.txt");
        exit(1);
    }
    int count;
    fscanf(points, "%d", &count);
    vertex *vertices = (vertex *)malloc(count * sizeof(vertex));
    parse(points, vertices, count);

    fclose(points);
*/
    