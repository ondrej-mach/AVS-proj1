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
constexpr int B_SIZE = 64;
constexpr int B_SIZE_SQ = (B_SIZE*B_SIZE);

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data = (int *)(aligned_alloc(ALIGN_BYTES, height * width * sizeof(int)));
	batchData = (int *)(aligned_alloc(ALIGN_BYTES, B_SIZE_SQ * sizeof(int)));;
    zRe = (float *)(aligned_alloc(ALIGN_BYTES, B_SIZE_SQ * sizeof(float)));
    zIm = (float *)(aligned_alloc(ALIGN_BYTES, B_SIZE_SQ * sizeof(float)));
	addRe = (float *)(aligned_alloc(ALIGN_BYTES, B_SIZE_SQ * sizeof(float)));
	addIm = (float *)(aligned_alloc(ALIGN_BYTES, B_SIZE_SQ * sizeof(float)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	free(data);
	free(batchData);
	free(zRe);
	free(zIm);
	free(addRe);
	free(addIm);

	data = NULL;
}


inline void BatchMandelCalculator::calculateBatch(const int batchX, const int batchY, const int batchWidth, const int batchHeight) {
	//#pragma omp simd
	for (int i=0; i<batchWidth; i++) {
		addRe[i] = x_start + (batchX+i) * dx;
	}

	//#pragma omp simd
	for (int i=0; i<batchHeight; i++) {
		addIm[i] = y_start + (batchY+i) * dy;
	}

	// Set initial values
	//#pragma omp simd collapse(2)
	for (int i=0; i<batchHeight; i++) {
		for (int j=0; j<batchWidth; j++) {
			zRe[i*batchWidth + j] = addRe[j];
			zIm[i*batchWidth + j] = addIm[i];
		}
	}

	for (int i = 0; i < B_SIZE_SQ; i++) {
		batchData[i] = limit;
	}

	// iterate the calculation for whole batch
	for (int i=0; i<limit; i++) {
		bool active = false;
		// Go through both coordinates of batch

		// workaround for omp aligned
		int * const data = this->data;
		int * const batchData = this->batchData;
		float * const zRe = this->zRe;
		float * const zIm = this->zIm;

		#pragma omp simd collapse(2) aligned(data, zRe, zIm:ALIGN_BYTES) safelen(64) reduction(||:active)
		for (int j=0; j<batchHeight; j++) {
			for (int k=0; k<batchWidth; k++) {

				int &result = batchData[j*batchWidth + k];

				if (result == limit) {
					float &re = zRe[j*batchWidth + k];
					float &im = zIm[j*batchWidth + k];
					float resq = re * re;
					float imsq = im * im;

					if (resq + imsq > 4.0f) {
						result = i;
					} else {
						float addRe = x_start + (batchX+k) * dx;
						float addIm = y_start + (batchY+j) * dy;
						im = 2.0f * re * im + addIm;
						re = resq - imsq + addRe;
						active = true;
					}
				}
			}
		}
		if (!active) {
			break;
		}
	}
}

int *BatchMandelCalculator::calculateMandelbrot() {
	int y = 0;
	while (y < (height+1)/2) {
		int batchHeight = std::min(height-y, B_SIZE);

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
		for (int j = 0; j < width; j++) {
			data[width*(height-i-1) + j] = data[width*i + j];
		}
	}

	return data;
}
