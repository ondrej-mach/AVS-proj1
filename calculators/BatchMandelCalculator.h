/**
 * @file BatchMandelCalculator.h
 * @author Ondřej Mach <xmacho12@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over small batches
 * @date 12. 10. 2023
 */
#ifndef BATCHMANDELCALCULATOR_H
#define BATCHMANDELCALCULATOR_H

#include <BaseMandelCalculator.h>

class BatchMandelCalculator : public BaseMandelCalculator
{
public:
    BatchMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~BatchMandelCalculator();
    int * calculateMandelbrot();

private:
    inline void calculateBatch(int batchX, int batchY, int batchWidth, int batchHeight);
    int *data;
    int *batchData;
    float *zRe;
    float *zIm;
    float *addRe;
    float *addIm;
};

#endif
