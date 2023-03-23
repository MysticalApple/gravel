#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double x;
    double y;
    double z;
} vertex;

int parse(FILE *points, vertex *vertices, int count) {
    for (int i = 0; i < count; i++) {
        fscanf(points, "%lf %lf %lf", &vertices[i].x, &vertices[i].y, &vertices[i].z);
    }
    return count;
}

int main(void) {
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

    points = fopen("cube.txt", "r");
    if (points == NULL) {
        perror("cube.txt");
        exit(1);
    }

    fscanf(points, "%d", &count);
    vertices = (vertex *)malloc(count * sizeof(vertex));
    parse(points, vertices, count);

    fclose(points);

    points = fopen("Pentpyr.txt", "r");
    if (points == NULL) {
        perror("Pentpyr.txt");
        exit(1);
    }

    fscanf(points, "%d", &count);
    vertices = (vertex *)malloc(count * sizeof(vertex));
    parse(points, vertices, count);

    fclose(points);

    points = fopen("tetrahedron.txt", "r");
    if (points == NULL) {
        perror("tetrahedron.txt");
        exit(1);
    }

    fscanf(points, "%d", &count);
    vertices = (vertex *)malloc(count * sizeof(vertex));
    parse(points, vertices, count);

    fclose(points);

    points = fopen("triprism.txt", "r");
    if (points == NULL) {
        perror("triprism.txt");
        exit(1);
    }

    fscanf(points, "%d", &count);
    vertices = (vertex *)malloc(count * sizeof(vertex));
    parse(points, vertices, count);

    fclose(points);

    for (int i = 0; i < count; i++) {
        vertex point = vertices[i];
        printf("x:%f   y:%f   z:%f\n", point.x, point.y, point.z);
    }

    free(vertices);
    return 0;
}