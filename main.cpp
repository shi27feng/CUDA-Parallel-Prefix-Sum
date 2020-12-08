#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include <math.h>

#include "SumPrescan.cu"

// Compares two arrays and outputs if they match or prints the first element that failed the check otherwise
bool compareArrays(int *array1, int *array2, int numElements) {
	for (int i = 0; i < numElements; ++i) {
		if (array1[i] != array2[i]) {
			printf("ARRAY CHECK FAIL at arr1 = %d, arr2 = %d, at index = %d\n", array1[i], array2[i], i);
			return false;
		}
	}
	return true;
}

int main(void) {
	// Error code to check return values for CUDA calls
	cudaError_t err = cudaSuccess;

	// For timing
	StopWatchInterface * timer = NULL;
	sdkCreateTimer(&timer);
	double h_msecs;

	// Number of elements in the array
	int numElements = 10000000;
	size_t size = numElements * sizeof(int);
    printf("Prescans of arrays of size %d:\n\n", numElements);

	int *h_x = (int *) malloc(size);
	int *h_yBlock = (int *) malloc(size);
	int *h_yFull = (int *) malloc(size);
	int *h_dOutput = (int *) malloc(size);

	if (h_x == NULL || h_yBlock == NULL || h_yFull == NULL || h_dOutput == NULL) {
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	unsigned int seed = 1;

	// Initialize the host array to random integers
	srand(seed);
	for (int i = 0; i < numElements; i++) {
		h_x[i] = rand() % 10;
	}

	//--------------------------Sequential Scans-------------------------------

	sdkStartTimer(&timer);
	hostBlockScan(h_x, h_yBlock, numElements);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("HOST sequential BLOCK scan on took = %.5fmSecs\n", h_msecs);

	sdkStartTimer(&timer);
	hostFullScan(h_x, h_yFull, numElements);
	sdkStopTimer(&timer);

	h_msecs = sdkGetTimerValue(&timer);
	printf("HOST squential FULL scan took = %.5fmSecs\n\n", h_msecs);

	//--------------------------Redo the input array---------------------------
	// Create a new identical host input array
	// This is needed because with large arrays (and only large arrays)
	// the hostBlockScan() method overrides some of the input array values.
	int *h_xNew = (int *) malloc(size);
	if (h_xNew == NULL) {
		fprintf(stderr, "Failed to allocate host vector!\n");
		exit(EXIT_FAILURE);
	}

	srand(seed);
	for (int i = 0; i < numElements; i++) {
		h_xNew[i] = rand() % 10;
	}

	//--------------------------Device Block Scans------------------------------

	// Create the device timer
	cudaEvent_t d_start, d_stop;
	float d_msecs;
	cudaEventCreate(&d_start);
	cudaEventCreate(&d_stop);

	int *d_x = NULL;
	err = cudaMalloc((void **) &d_x, size);
	CUDA_ERROR(err, "Failed to allocate device array x");

	int *d_y = NULL;
	err = cudaMalloc((void**) &d_y, size);
	CUDA_ERROR(err, "Failed to allocate device array y");

	err = cudaMemcpy(d_x, h_xNew, size, cudaMemcpyHostToDevice);
	CUDA_ERROR(err, "Failed to copy array xNew from host to device");

	// Blocks per grid for the block scans
	int blocksPerGrid = 1 + ((numElements - 1) / (BLOCK_SIZE * 2));

	//----------------------Device Non BCAO Block Scan-------------------------

	cudaEventRecord(d_start, 0);
	blockPrescan<<<blocksPerGrid, BLOCK_SIZE>>>(d_x, d_y, numElements, NULL);
	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);

	// Wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch blockPrescan kernel");

	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	err = cudaMemcpy(h_dOutput, d_y, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array y from device to host");

	// Verify that the result vector is correct
//	printf("BLOCK non-BCAO prescan took %.5f msecs\n", d_msecs);

	if(compareArrays(h_dOutput, h_yBlock, numElements)){
		printf("DEVICE BLOCK non-BCAO prescan test passed, the scan took %.5f msecs\n", d_msecs);
	}else{
		printf("DEVICE BLOCK non-BCAO prescan test failed, the scan took %.5f msecs\n", d_msecs);
	}

	//----------------------Device BCAO Block Scan-----------------------------

	cudaEventRecord(d_start, 0);
	BCAO_blockPrescan<<<blocksPerGrid, BLOCK_SIZE>>>(d_x, d_y, numElements, NULL);
	cudaEventRecord(d_stop, 0);
	cudaEventSynchronize(d_stop);

	// Wait for device to finish
	cudaDeviceSynchronize();

	err = cudaGetLastError();
	CUDA_ERROR(err, "Failed to launch BCAO_blockPrescan kernel");

	err = cudaEventElapsedTime(&d_msecs, d_start, d_stop);
	CUDA_ERROR(err, "Failed to get elapsed time");

	err = cudaMemcpy(h_dOutput, d_y, size, cudaMemcpyDeviceToHost);
	CUDA_ERROR(err, "Failed to copy array y from device to host");

	// Verify that the result vector is correct
//	printf("BLOCK BCAO prescan took %.5f msecs\n", d_msecs);

	if(compareArrays(h_dOutput, h_yBlock, numElements)){
		printf("DEVICE BLOCK BCAO prescan test passed, the scan took %.5f msecs\n\n", d_msecs);
	}else{
		printf("DEVICE BLOCK BCAO prescan test failed, the scan took %.5f msecs\n\n", d_msecs);
	}

	// Free device memory as full scan methods will allocate their own memory
	err = cudaFree(d_x);
	CUDA_ERROR(err, "Failed to free device array x");
	err = cudaFree(d_y);
	CUDA_ERROR(err, "Failed to free device array y");

	//--------------------------Device Full Scans------------------------------

	fullPrescan(h_x, h_yFull, numElements);

	BCAO_fullPrescan(h_x, h_yFull, numElements);

	//--------------------------Cleanup----------------------------------------

	// Destroy device timer events
	cudaEventDestroy(d_start);
	cudaEventDestroy(d_stop);

	// Delete host timer
	sdkDeleteTimer(&timer);

	// Reset the device
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

	// Free host memory
	free(h_x);
	free(h_yBlock);
	free(h_yFull);
	free(h_dOutput);

	printf("\nFinished");

	return 0;
}
