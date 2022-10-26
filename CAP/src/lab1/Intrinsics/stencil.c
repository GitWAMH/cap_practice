#include <immintrin.h>
#include "stencil.h"
#include <stdio.h>

void ApplyStencil(unsigned char *img_in, unsigned char *img_out, int width, int height) {

	__m256i val, val_out;
	
	for (int i = 1; i < height-1; i++) 

		for (int j = 1; j < width-1; j+=8) {

			__m256i rA = _mm256_load_si256((const __m256i *) &img_in[((i-1)*width + j-1)]);
			__m256i rB = _mm256_load_si256((const __m256i *) &img_in[((i-1)*width + j)]);
			__m256i rC = _mm256_load_si256((const __m256i *) &img_in[((i-1)*width + j+1)]);
			
			__m256i rD = _mm256_load_si256((const __m256i *) &img_in[((i  )*width + j-1)]);
			__m256i rE = _mm256_load_si256((const __m256i *) &img_in[((i  )*width + j)]);
			__m256i rF = _mm256_load_si256((const __m256i *) &img_in[((i  )*width + j+1)]);

			__m256i rG = _mm256_load_si256((const __m256i *) &img_in[((i+1)*width + j-1)]);
			__m256i rH = _mm256_load_si256((const __m256i *) &img_in[((i+1)*width + j)]);
			__m256i rI = _mm256_load_si256((const __m256i *) &img_in[((i+1)*width + j+1)]);
			
			__m256i mul7 = _mm256_adds_epu8(rE, _mm256_adds_epu8(rE, _mm256_adds_epu8(rE, _mm256_adds_epu8(rE, _mm256_adds_epu8(rE, _mm256_adds_epu8(rE, rE))))));
			
			val = rE;

			val = _mm256_subs_epu8(val,rA);
			val = _mm256_subs_epu8(val,rB);
			val = _mm256_subs_epu8(val,rC);

			val = _mm256_subs_epu8(val,rD);
			val = _mm256_adds_epu8(val, mul7);
			val = _mm256_subs_epu8(val,rF);

			val = _mm256_subs_epu8(val,rG);
			val = _mm256_subs_epu8(val,rH);
			val = _mm256_subs_epu8(val,rI);

			_mm256_store_si256((__m256i *) &(img_out[((i  )*width + j)]), val);
		}

}

void CopyImage(unsigned char *img_in, unsigned char *img_out, int width, int height) {
	for (int i = 0; i < height; i++)
		#pragma omp simd
		for (int j = 0; j < width; j++)
			img_in[i*width + j] = img_out[i*width + j];
}
