#include "cuda_runtime.h"

extern cudaError_t calcDistanceCoordiantesWithCuda(float* coordinates_1, float* coordinates_2, float* coordinates_arr, int num_coordinates);