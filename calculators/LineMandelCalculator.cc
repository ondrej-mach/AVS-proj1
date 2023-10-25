/**
 * @file LineMandelCalculator.cc
 * @author Ondřej Mach <xmacho12@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 12. 10. 2023
 */
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <stdlib.h>

#include "LineMandelCalculator.h"


constexpr int ALIGN_BYTES = 64;

LineMandelCalculator::LineMandelCalculator (unsigned matrixBaseSize, unsigned limit) :
	BaseMandelCalculator(matrixBaseSize, limit, "LineMandelCalculator")
{
	data = (int *)(aligned_alloc(ALIGN_BYTES, height*width*sizeof(int)));
	lineReal = (float *)(aligned_alloc(ALIGN_BYTES, width*sizeof(float)));
	lineImag = (float *)(aligned_alloc(ALIGN_BYTES, width*sizeof(float)));
}

LineMandelCalculator::~LineMandelCalculator() {
	free(data);
	free(lineReal);
	free(lineImag);

	data = NULL;
}


void LineMandelCalculator::calculateLine(int lineNumber)
{
	float imagAdd = y_start + lineNumber * dy;

	// Set initial values for whole line
	for (int i=0; i<width; i++) {
		lineReal[i] = x_start + i * dx;
		lineImag[i] = imagAdd;
	}

	// Iterate for whole line
	for (int i=0; i<limit; i++) {
		// for all complex number in line
		bool skip = true;

		int * const data = this->data;
		float * const lineReal = this->lineReal;
		float * const lineImag = this->lineImag;

		#pragma omp simd aligned(data, lineReal, lineImag:ALIGN_BYTES)
		for (int j=0; j<width; j++) {
			// If point has not diverged yet
			if (data[lineNumber*width + j] == limit) {
				float &re = lineReal[j];
				float &im = lineImag[j];
				float resq = re * re;
				float imsq = im * im;

				if (resq + imsq > 4.0f) {
					data[lineNumber*width + j] = i;
				} else {
					float realAdd = x_start + j*dx;
					im = 2.0f * re * im + imagAdd;
					re = resq - imsq + realAdd;
					skip = false;
				}
			}
		}
		if (skip) {
			break;
		}
	}
}


int * LineMandelCalculator::calculateMandelbrot () {
	for (int i = 0; i < width*height; i++) {
		data[i] = limit;
	}
	for (int i = 0; i < (height+1)/2; i++) {
		calculateLine(i);
	}
	// Copy lines to lower half (symmetrically along x axis)
	for (int i = 0; i < height/2; i++) {
		for (int j = 0; j < width; j++) {
			data[width*(height-i-1) + j] = data[width*i + j];
		}
	}
	return data;
}
