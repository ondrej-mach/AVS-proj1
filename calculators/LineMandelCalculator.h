/**
 * @file LineMandelCalculator.h
 * @author Ondřej Mach <xmacho12@stud.fit.vutbr.cz>
 * @brief Implementation of Mandelbrot calculator that uses SIMD paralelization over lines
 * @date 12. 10. 2023
 */

#include <BaseMandelCalculator.h>

class LineMandelCalculator : public BaseMandelCalculator
{
public:
    LineMandelCalculator(unsigned matrixBaseSize, unsigned limit);
    ~LineMandelCalculator();
    int *calculateMandelbrot();

private:
    inline void calculateLine(int lineNumber);
    int *data;
    float *lineReal;
    float *lineImag;
};
