#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

__global__ void increment_kernel(int* g_data, int inc_value)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	g_data[idx] = g_data[idx] + inc_value;
}

__global__ void kernel_matrix_multiply(float* P, float* M, float* N, int dimSize) {
	int col = blockIdx.y * blockDim.y + threadIdx.y;
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	float sum = 0.0f;
	if (row < dimSize && col < dimSize) {
		for (int k = 0; k < dimSize; ++k) {
			sum += M[row * dimSize + k] * N[k * dimSize + col];
		}
		P[row * dimSize + col] = sum;
	}
}

void host_matrix_multiply(float* P, float* M, float* N, int dimSize) {
	dim3 blockSize(16, 16);
	dim3 gridSize((dimSize + blockSize.x - 1) / blockSize.x, (dimSize + blockSize.y - 1) / blockSize.y);

	// kernel
	kernel_matrix_multiply << <gridSize, blockSize >> > (P, M, N, dimSize);
	cudaDeviceSynchronize();
}

int correct_output(int* data, const int n, const int x)
{
	for (int i = 0; i < n; i++)
		if (data[i] != x)
			return 0;
	return 1;
}

int getNumCores(cudaDeviceProp devProp)
{
	int cores = 0;
	int mp = devProp.multiProcessorCount;
	switch (devProp.major) {
	case 2: // Fermi
		if (devProp.minor == 1) cores = mp * 48;
		else cores = mp * 32;
		break;
	case 3: // Kepler
		cores = mp * 192;
		break;
	case 5: // Maxwell
		cores = mp * 128;
		break;
	case 6: // Pascal
		if ((devProp.minor == 1) || (devProp.minor == 2)) cores = mp * 128;
		else if (devProp.minor == 0) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 7: // Volta and Turing
		if ((devProp.minor == 0) || (devProp.minor == 5)) cores = mp * 64;
		else printf("Unknown device type\n");
		break;
	case 8: // Ampere
		if (devProp.minor == 0) cores = mp * 64;
		else if (devProp.minor == 6) cores = mp * 128;
		else if (devProp.minor == 9) cores = mp * 128; // ada lovelace
		else printf("Unknown device type\n");
		break;
	case 9: // Hopper
		if (devProp.minor == 0) cores = mp * 128;
		else printf("Unknown device type\n");
		break;
	default:
		printf("Unknown device type\n");
		break;
	}
	return cores;
}

int main(int argc, char* argv[])
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	printf("Number of CUDA devices: %d\n", deviceCount);

	// loop over and check every device
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp property;
		cudaGetDeviceProperties(&property, i);

		// print out each device property.
		printf("Device name: %s\n", property.name);
		printf("Clock rate: %d MHz\n", property.clockRate/1000);
		printf("Number of streaming multiprocessors: %d\n", property.multiProcessorCount);
		printf("Number of cores: %d\n", getNumCores(property));
		printf("Warp size: %d\n", property.warpSize);
		printf("Global memory: %.2f GB\n", (float)property.totalGlobalMem / (1024 * 1024) / 1000);
		printf("Constant memory: %.2f KB\n", (float)property.totalConstMem / 1024);
		printf("Shared memory per block: %.2f KB\n", (float)property.sharedMemPerBlock / 1024);
		printf("Registers available per block: %d\n", property.regsPerBlock);
		printf("Maximum threads per block: %d\n", property.maxThreadsPerBlock);
		printf("Maximum block dimensions: %d x %d x %d\n", property.maxThreadsDim[0], property.maxThreadsDim[1], property.maxThreadsDim[2]);
		printf("Maximum grid dimensions: %d x %d x %d\n", property.maxGridSize[0], property.maxGridSize[1], property.maxGridSize[2]);
	}


	int dimSize = 1024;

	// allocate matrix memory on host
	float* M = (float*)malloc(dimSize * dimSize * sizeof(float));
	float* N = (float*)malloc(dimSize * dimSize * sizeof(float));
	float* P = (float*)malloc(dimSize * dimSize * sizeof(float));
	float* P_cpu = (float*)malloc(dimSize * dimSize * sizeof(float)); // For CPU-computed result

	// Initialize matrices M and N with random values
	for (int i = 0; i < dimSize * dimSize; ++i) {
		M[i] = rand() / (float)RAND_MAX;
		N[i] = rand() / (float)RAND_MAX;
	}

	// Allocate memory for matrices on device
	float* d_M, * d_N, * d_P;
	cudaMalloc(&d_M, dimSize * dimSize * sizeof(float));
	cudaMalloc(&d_N, dimSize * dimSize * sizeof(float));
	cudaMalloc(&d_P, dimSize * dimSize * sizeof(float));

	// Transfer input matrices from host to device
	cudaMemcpy(d_M, M, dimSize * dimSize * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_N, N, dimSize * dimSize * sizeof(float), cudaMemcpyHostToDevice);

	// Perform matrix multiplication on device
	host_matrix_multiply(d_P, d_M, d_N, dimSize);

	// Transfer output matrix from device to host
	cudaMemcpy(P, d_P, dimSize * dimSize * sizeof(float), cudaMemcpyDeviceToHost);

	// Compute matrix multiplication on CPU for comparison
	for (int i = 0; i < dimSize; ++i) {
		for (int j = 0; j < dimSize; ++j) {
			float sum = 0.0f;
			for (int k = 0; k < dimSize; ++k) {
				sum += M[i * dimSize + k] * N[k * dimSize + j];
			}
			P_cpu[i * dimSize + j] = sum;
		}
	}

	// Check for correctness within a certain tolerance
	float tolerance = 1e-5;
	bool passed = true;
	for (int i = 0; i < dimSize * dimSize; ++i) {
		if (fabs(P[i] - P_cpu[i]) > tolerance) {
			passed = false;
			break;
		}
	}

	if (passed) {
		printf("Test PASSED\n");
	}
	else {
		printf("Test FAILED\n");
	}

	// Free memory on host and device
	free(M);
	free(N);
	free(P);
	free(P_cpu);
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P);
	return 0;
}