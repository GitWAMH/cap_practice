#define _GNU_SOURCE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>

#include "colormap.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// Simulation parameters
static const unsigned int N = 2048;

static const float SOURCE_TEMP   = 100.0f;
static const float ENVIROM_TEMP  = 25.0f;
static const float BOUNDARY_TEMP = 5.0f;

static const float MIN_DELTA = 0.01f;
static const unsigned int MAX_ITERATIONS = 2000;



static void init(unsigned int source_x, unsigned int source_y, float * matrix) {
    // init
    //#pragma omp parallel for collapse(2)
    for (unsigned int y = 0; y < N; ++y)
        for (unsigned int x = 0; x < N; ++x)
            matrix[y*N+x]=ENVIROM_TEMP;

    // place source
    matrix[source_y*N+source_x] = SOURCE_TEMP;

    // fill borders
    for (unsigned int x = 0; x < N; ++x) {
        matrix[        x] = BOUNDARY_TEMP;
        matrix[(N-1)*N+x] = BOUNDARY_TEMP;
    }
    for (unsigned int y = 0; y < N; ++y) {
        matrix[y*N    ] = BOUNDARY_TEMP;
        matrix[y*N+N-1] = BOUNDARY_TEMP;
    }
}

static void step(unsigned int source_x, unsigned int source_y, const float *restrict current, float *restrict next) {

    #pragma omp target enter data map(to: source_x, source_y, current, N, next) map (alloc: next)
    { 
        float a = 0.5f; //Diffusion constant
        float dx = 0.01f; float dx2 = dx*dx;
        float dy = 0.01f; float dy2 = dy*dy;

        float dt = dx2 * dy2 / (2.0 * a * (dx2 + dy2));
        
        #pragma omp target teams distribute parallel for collapse(2) schedule (guided)
        #pragma ivdep
        for (unsigned y = 1; y < N-1; ++y) {
            for (unsigned x = 1; x < N-1; ++x) {
                next[y*N+x] = current[y*N+x] + a * dt *
                    ((current[y*N+x+1]   - 2.0*current[y*N+x] + current[y*N+x-1])/dx2 +
                    (current[(y+1)*N+x] - 2.0*current[y*N+x] + current[(y-1)*N+x])/dy2);
            }
        }

        int is_cpu = omp_is_initial_device();
		printf("Running on %s\n", is_cpu ? "CPU" : "GPU");
        #pragma omp target
        next[source_y*N+source_x] = SOURCE_TEMP;

        #pragma omp target exit data map (from:next[0:N * N * sizeof(float)])
    }
}

static float diff(const float *restrict current, const float *restrict next) {
    float maxdiff = 0.0f;
    #pragma omp target enter data map(to: current, next, maxdiff, N) 
    {
        
        int is_cpu = omp_is_initial_device();
		printf("Running on %s\n", is_cpu ? "CPU" : "GPU");

        #pragma omp target teams distribute parallel for collapse (2) schedule (guided) reduction (max:maxdiff)
        #pragma ivdep
        for (unsigned int y = 1; y < N-1; ++y) {
            for (unsigned int x = 1; x < N-1; ++x) {
                maxdiff = fmaxf(maxdiff, fabsf(next[y*N+x] - current[y*N+x]));
            }
        }
        #pragma omp target update from (maxdiff)

        #pragma omp target exit data map (release: maxdiff)
    }
    return maxdiff;
}


void write_png(float * current, int iter) {
    char file[100];
    uint8_t * image = malloc(3 * N * N * sizeof(uint8_t));
    float maxval = fmaxf(SOURCE_TEMP, BOUNDARY_TEMP);

    //#pragma omp parallel for collapse(2)
    for (unsigned int y = 0; y < N; ++y) {
        for (unsigned int x = 0; x < N; ++x) {
            unsigned int i = y*N+x;
            colormap_rgb(COLORMAP_MAGMA, current[i], 0.0f, maxval, &image[3*i], &image[3*i + 1], &image[3*i + 2]);
        }
    }
    sprintf(file,"heat%i.png", iter);
    stbi_write_png(file, N, N, 3, image, 3 * N);

    free(image);
}


int main() {
    size_t array_size = N * N * sizeof(float);

    float * current = malloc(array_size);
    float * next = malloc(array_size);

    srand(0);
    unsigned int source_x = rand() % (N-2) + 1;
    unsigned int source_y = rand() % (N-2) + 1;
    unsigned int it = 0;

    printf("Heat source at (%u, %u)\n", source_x, source_y);

    init(source_x, source_y, current);
    init(source_x, source_y, next);

    double start = omp_get_wtime();

    float t_diff = SOURCE_TEMP;
    for (it = 0; (it < MAX_ITERATIONS) && (t_diff > MIN_DELTA); ++it) {
        step(source_x, source_y, current, next);
        t_diff = diff(current, next);
        if(it%(MAX_ITERATIONS/40)==0){
            printf("%u: %f\n", it, t_diff);
        }

        float * swap = current;
        current = next;
        next = swap;
    }
    double stop = omp_get_wtime();
    printf("Computing time %f s.\n", stop-start);

    write_png(current, it);

    free(current);
    free(next);

    return 0;
}
