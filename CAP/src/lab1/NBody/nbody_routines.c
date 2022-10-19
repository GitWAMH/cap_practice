#include <math.h>
#include "nbody.h"

void bodyForce(body *p, float dt, int n) {

	float softeningSquared = 1e-3;
	float G = 6.674e-11;

	for (int i = 0; i < n; i++) { 
		float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

		#pragma vector aligned
		for (int j = 0; j < i-1; j++) {
			float dx = p->x[j] - p->x[i];
			float dy = p->y[j] - p->y[i];
			float dz = p->z[j] - p->z[i];
			float distSqr = dx*dx + dy*dy + dz*dz + softeningSquared;
			float invDist = 1.0f / sqrtf(distSqr);
			float invDist3 = invDist * invDist * invDist;

			float g_masses = G * p->m[j] * p->m[i];

			Fx += g_masses * dx * invDist3; 
			Fy += g_masses * dy * invDist3; 
			Fz += g_masses * dz * invDist3;
		}
		
		#pragma vector aligned
		for (int k = i+1; k < n; k++){
			float dx = p->x[k] - p->x[i];
			float dy = p->y[k] - p->y[i];
			float dz = p->z[k] - p->z[i];
			float distSqr = dx*dx + dy*dy + dz*dz + softeningSquared;
			float invDist = 1.0f / sqrtf(distSqr);
			float invDist3 = invDist * invDist * invDist;

			float g_masses = G * p->m[k] * p->m[i];

			Fx += g_masses * dx * invDist3; 
			Fy += g_masses * dy * invDist3; 
			Fz += g_masses * dz * invDist3;
		}

		p->vx[i] += dt*Fx/p->m[i]; 
		p->vy[i] += dt*Fy/p->m[i]; 
		p->vz[i] += dt*Fz/p->m[i];
	}

}

void integrate(body *p, float dt, int n){
	#pragma ivdep
	#pragma vector aligned
	for (int i = 0 ; i < n; i++) {
		p->x[i] += p->vx[i]*dt;
		p->y[i] += p->vy[i]*dt;
		p->z[i] += p->vz[i]*dt;
	}
}
