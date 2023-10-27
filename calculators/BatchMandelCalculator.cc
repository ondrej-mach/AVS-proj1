/**
 * @file BatchMandelCalculator.cc
 * @author Ondřej Mach <xmacho12@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 12. 10. 2023
 */

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>
#include <stdexcept>

#include "BatchMandelCalculator.h"


constexpr int ALIGN_BYTES = 64;
constexpr int B_PWR = 6;
constexpr int B_SIZE = 1 << B_PWR;
constexpr int B_SIZE_SQ = (B_SIZE*B_SIZE);

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data = (int *)(aligned_alloc(ALIGN_BYTES, height * width * sizeof(int)));
	batchData = (int *)(aligned_alloc(ALIGN_BYTES, B_SIZE_SQ * sizeof(int)));;
    zRe = (float *)(aligned_alloc(ALIGN_BYTES, B_SIZE_SQ * sizeof(float)));
    zIm = (float *)(aligned_alloc(ALIGN_BYTES, B_SIZE_SQ * sizeof(float)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	free(data);
	free(batchData);
	free(zRe);
	free(zIm);

	data = NULL;
}


inline void BatchMandelCalculator::calculateBatch(const int batchX, const int batchY, int batchWidth, int batchHeight) {
	#if defined(__INTEL_COMPILER)
	__assume_aligned(this->data, ALIGN_BYTES);
	__assume_aligned(this->batchData, ALIGN_BYTES);
	__assume_aligned(this->zRe, ALIGN_BYTES);
	__assume_aligned(this->zIm, ALIGN_BYTES);
	#endif

	// convert from double to float
	const float x_start = this->x_start;
	const float y_start = this->y_start;
	const float dx = this->dx;
	const float dy = this->dy;

	// Better optimisation if we compute entire square
	// instead of rectangle limited by function's parameters
	const int bw = B_SIZE;
	const int bh = B_SIZE;

	// Set initial values
	#pragma omp simd
	for (int i=0; i<B_SIZE_SQ; i++) {
		zRe[i] = x_start + (batchX + (i%bw)) * dx;
		zIm[i] = y_start + (batchY + (i/bw)) * dy;
	}

	for (int i = 0; i < B_SIZE_SQ; i++) {
		batchData[i] = limit;
	}

	for (int bline = 0; bline < bh; bline++) {
		// iterate the calculation for a line in batch
		for (int i=0; i<limit; i++) {
			bool active = false;

			#pragma omp simd reduction(||:active)
			for (int bcol=0; bcol<bw; bcol++) {
				int &result = batchData[bline*bw + bcol];
				if (result == limit) {
					float &re = zRe[bline*bw + bcol];
					float &im = zIm[bline*bw + bcol];
					float resq = re * re;
					float imsq = im * im;

					if (resq + imsq > 4.0f) {
						result = i;
					} else {
						float addRe = x_start + (batchX + bcol) * dx;
						float addIm = y_start + (batchY + bline) * dy;
						im = 2.0f * re * im + addIm;
						re = resq - imsq + addRe;
						active = true;
					}
				}
			}
			if (!active) {
				break;
			}
		}
	}

	// copy data from batch buffer to global
	for (int j=0; j<batchHeight; j++) {
		//#pragma omp simd
        for (int k=0; k<batchWidth; k++) {
			data[(batchY+j)*width + (batchX+k)] = batchData[j*batchWidth + k];
		}
	}
}

int *BatchMandelCalculator::calculateMandelbrot() {
	#if defined(__INTEL_COMPILER)
	__assume_aligned(this->data, ALIGN_BYTES);
	__assume_aligned(this->batchData, ALIGN_BYTES);
	__assume_aligned(this->zRe, ALIGN_BYTES);
	__assume_aligned(this->zIm, ALIGN_BYTES);
	#endif

	int y = 0;
	while (y < (height+1)/2) {
		int batchHeight = std::min((height+1)/2, B_SIZE);

		int x = 0;
		while (x < width) {
			int batchWidth = std::min(width-x, B_SIZE);
			calculateBatch(x, y, batchWidth, batchHeight);
			x += batchWidth;
		}

		x = 0;
		y += batchHeight;
	}

	// Copy lines to lower half (symmetrically along x axis)
	for (int i = 0; i < height/2; i++) {
		#pragma omp simd
		for (int j = 0; j < width; j++) {
			data[width*(height-i-1) + j] = data[width*i + j];
		}
	}

	return data;
}
