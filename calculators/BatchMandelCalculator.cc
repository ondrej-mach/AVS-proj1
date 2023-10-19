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

// TODO
#define B_SIZE 64
#define B_SIZE_SQ (B_SIZE*B_SIZE)

BatchMandelCalculator::BatchMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "BatchMandelCalculator")
{
	data = (int *)(malloc(height * width * sizeof(int)));;
    zRe = (float *)(malloc(B_SIZE_SQ * sizeof(float)));
    zIm = (float *)(malloc(B_SIZE_SQ * sizeof(float)));
	addRe = (float *)(malloc(B_SIZE_SQ * sizeof(float)));
	addIm = (float *)(malloc(B_SIZE_SQ * sizeof(float)));
}

BatchMandelCalculator::~BatchMandelCalculator() {
	free(data);
	free(zRe);
	free(zIm);
	free(addRe);
	free(addIm);

	data = NULL;
}

// batchWidth, batchHeight <= B_SIZE
#pragma omp declare simd
inline void BatchMandelCalculator::calculateBatch(int batchX, int batchY, int batchWidth, int batchHeight) {
	for (int i=0; i<batchWidth; i++) {
		addRe[i] = x_start + (batchX+i) * dx;
	}

	for (int i=0; i<batchHeight; i++) {
		addIm[i] = y_start + (batchY+i) * dy;
	}

	// Set initial values
	for (int i=0; i<batchHeight; i++) {
		for (int j=0; j<batchWidth; j++) {
			zRe[i*batchWidth + j] = addRe[j];
			zIm[i*batchWidth + j] = addIm[i];
		}
	}

	// iterate the calculation for whole batch
	for (int i=0; i<limit; i++) {
		bool skip = true;
		// Go through both coordinates of batch

		#pragma omp simd reduction(&&:skip)
		for (int j=0; j<batchHeight; j++) {
			for (int k=0; k<batchWidth; k++) {

				int &result = data[(batchY+j)*width + batchX+k];

				if (result == limit) {
					float &re = zRe[j*batchWidth + k];
					float &im = zIm[j*batchWidth + k];
					float resq = re * re;
					float imsq = im * im;

					if (resq + imsq > 4.0f) {
						result = i;
					} else {
						im = 2.0f * re * im + addIm[j];
						re = resq - imsq + addRe[k];
						skip = false;
					}
				}

			}
		}
		if (skip) {
			break;
		}
	}
}

int *BatchMandelCalculator::calculateMandelbrot() {
	for (int i = 0; i < width*height; i++) {
		data[i] = limit;
	}

	int y = 0;
	while (y<height) {
		int batchHeight = std::min(height-y, B_SIZE);

		int x = 0;
		while (x<width) {
			int batchWidth = std::min(width-x, B_SIZE);
			calculateBatch(x, y, batchWidth, batchHeight);
			x += batchWidth;
		}

		x = 0;
		y += batchHeight;
	}

	return data;
}
