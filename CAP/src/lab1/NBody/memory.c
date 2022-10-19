
#include <malloc.h>
#include <stdlib.h>
#include "nbody.h"

const int alignment = 32;

body* get_memory(int nBodies)
{
	body *p = (body*)_mm_malloc(nBodies*sizeof(body), alignment);
	p->m = (float *)_mm_malloc(nBodies*sizeof(float), alignment);
	p->x = (float *)_mm_malloc(nBodies*sizeof(float), alignment);
	p->y = (float *)_mm_malloc(nBodies*sizeof(float), alignment);
	p->z = (float *)_mm_malloc(nBodies*sizeof(float), alignment);
	p->vx = (float *)_mm_malloc(nBodies*sizeof(float), alignment);
	p->vy = (float *)_mm_malloc(nBodies*sizeof(float), alignment);
	p->vz = (float *)_mm_malloc(nBodies*sizeof(float), alignment);
	
	return p;
}

void free_memory(body *p)
{
	_mm_free(p->m);
	_mm_free(p->x);
	_mm_free(p->y);
	_mm_free(p->z);
	_mm_free(p->vx);
	_mm_free(p->vy);
	_mm_free(p->vz);
	
	_mm_free(p);
}

void randomizeBodies(body *data, int n) {
	#pragma ivdep
	for (int i = 0; i < n; i++) {
		data->m[i]  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		data->x[i]  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data->y[i]  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data->z[i]  = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

		data->vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data->vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
		data->vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
	}
}

