//Program for performing GPU-accelerated single-photon peak event detection for FLIM
//By Janet Sorrells and Rishee Iyer, 2021
//Contact: janetes2@illinois.edu
//These DLL funcitons can be called by LabView for real-time processing!

#include "KernelFile.cuh"

uint16_t* GPU_FLIMRawData_1Chunk;
float* GPU_numXxnumY_MeanLifetime_AfterAv, * GPU_numXxnumY_MPMImage_BeforeAv;
uint16_t* GPU_numXxnumY_S_AfterAv, * GPU_numXxnumY_G_AfterAv;
uint8_t* GPU_Thresh_Array;
float* gThresh;
int32_t* gnumX, * gnumY, * gnumTTot, * gnumT, * gDivFactor, * gNumChunks, * gChunkIndex, * gnumYPerChunk, * gnumXnumYnumT, * gNumxidMax;
CountDataType* GPU_numXxnumYxnumTxnumPulses_Count_Container;
AvgDataType* GPU_numXxnumYxnumT_Container, * GPU_numXxnumYxnumT_Container_Shifted;

// Initialize GPU to free memeroy and select device
DLLEXPORT cudaError_t FastFLIMGPU_StartGPU()
{

	FILE* LogFile;
	if (FILESTUFFEH) LogFile = fopen(FLIMLOGFILENAME, "wt");
	if (FILESTUFFEH) fprintf(LogFile, "START GPU\n");
	int deviceCount = 0;
	cudaError_t CC;

	cudaGetDeviceCount(&deviceCount);
	for (int i = 0; i < 1; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		CC = cudaSetDevice(i);
		if (FILESTUFFEH) fprintf(LogFile, "I selected this device: %s. You cool with it, champ?\n\n", prop.name);

	}
	CC = cudaGetLastError();
	if (FILESTUFFEH) fprintf(LogFile, "Last Cuda Error: %s\n", cudaGetErrorString(cudaPeekAtLastError()));
	if (FILESTUFFEH) fclose(LogFile);
	return (CC);
}


//Calculate array size, allocate memory, and copy data to device
DLLEXPORT cudaError_t FastFLIMGPU_InitializeAndAllocateMemory(int32_t* numX, int32_t* numY, int32_t* numTTot, int32_t* numT, int32_t* numChunks)
{
	FILE* LogFile;
	if (FILESTUFFEH) fopen_s(&LogFile, FLIMLOGFILENAME, "w");
	if (FILESTUFFEH) fprintf(LogFile, "INITIALIZATION STEP\n");

	//Select the device and reset it
	cudaSetDevice(0);
	cudaDeviceReset();

	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Device reset failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	//Allocate memory for array sizes and indices
	cudaMalloc((void**)& gnumX, sizeof(int32_t));
	cudaMalloc((void**)& gnumY, sizeof(int32_t));
	cudaMalloc((void**)& gnumT, sizeof(int32_t));
	cudaMalloc((void**)& gnumTTot, sizeof(int32_t));
	cudaMalloc((void**)& gDivFactor, sizeof(int32_t));
	cudaMalloc((void**)& gNumChunks, sizeof(int32_t));
	cudaMalloc((void**)& gChunkIndex, sizeof(int32_t));
	cudaMalloc((void**)& gnumYPerChunk, sizeof(int32_t));
	cudaMalloc((void**)& gThresh, sizeof(float));
	cudaMalloc((void**)& gnumXnumYnumT, sizeof(int32_t));
	cudaMalloc((void**)& gNumxidMax, sizeof(int32_t));

	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Malloc 1 failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	int32_t numYPerChunks = *numY / *numChunks;

	//Allocate memory for data containers
	cudaMalloc((void**)&GPU_FLIMRawData_1Chunk, sizeof(uint16_t) * *numX * numYPerChunks * *numTTot);
	cudaMalloc((void**)&GPU_numXxnumYxnumT_Container, sizeof(AvgDataType) * *numX * *numY * *numT);
	cudaMalloc((void**)&GPU_numXxnumYxnumT_Container_Shifted, sizeof(AvgDataType) * *numX * *numY * *numT);
	cudaMalloc((void**)&GPU_numXxnumY_MeanLifetime_AfterAv, sizeof(float) * *numX * *numY);
	cudaMalloc((void**)&GPU_numXxnumY_MPMImage_BeforeAv, sizeof(float) * *numX * *numY);
	cudaMalloc((void**)&GPU_numXxnumY_S_AfterAv, sizeof(uint16_t) * *numX * *numY);
	cudaMalloc((void**)&GPU_numXxnumY_G_AfterAv, sizeof(uint16_t) * *numX * *numY);
	cudaMalloc((void**)&GPU_Thresh_Array, sizeof(uint8_t) * *numX * *numY);
	cudaMalloc((void**)&GPU_numXxnumYxnumTxnumPulses_Count_Container, sizeof(CountDataType) * *numX * numYPerChunks * *numTTot);

	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Malloc 2 failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	//Array size calculations
	int32_t DivFactor = *numTTot / *numT;
	int32_t xyt = *numX * *numY * *numT;
	int32_t xyttot = *numX * numYPerChunks * *numTTot - 2;

	//Copy over array sizes
	cudaMemcpy(gnumX, numX, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gnumY, numY, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gnumT, numT, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gnumTTot, numTTot, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gNumChunks, numChunks, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gnumYPerChunk, &numYPerChunks, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gDivFactor, &DivFactor, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gnumXnumYnumT, &xyt, sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaMemcpy(gNumxidMax, &xyttot, sizeof(int32_t), cudaMemcpyHostToDevice);

	if (FILESTUFFEH) fprintf(LogFile, "Parameters | X: %d, Y: %d, TTot: %d, T: %d, C: %d, YpC: %d, DivFactor: %d\n", 
		*numX, *numY, *numTTot, *numT, *numChunks, numYPerChunks, DivFactor);

	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Memcpy failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	if (FILESTUFFEH) fprintf(LogFile, "ANYTHING THAT HAPPENS AFTER THIS IS IN THE PROCESSING STEP\n");
	if (FILESTUFFEH) fclose(LogFile);

	return cudaPeekAtLastError();
}

//Count photons and histogram counts for one chunk
DLLEXPORT cudaError_t FastFLIMGPU_ReportFrame(uint16_t* RawData_Block, int32_t numX, int32_t numYPerChunk, int32_t numTTot, int32_t numT,
	int32_t numChunks, int32_t ChunkIndex)
{
	FILE* LogFile;
	if (FILESTUFFEH) fopen_s(&LogFile, FLIMLOGFILENAME, "a"); 
	dim3 dimBlock3D(BLOCK_SIZE_4A, BLOCK_SIZE_4A, BLOCK_SIZE_T); 
	dim3 dimGrid3D((numX) / dimBlock3D.x, (numYPerChunk) / dimBlock3D.y, (numT) / dimBlock3D.z);
	dim3 dimBlock1D(numX/2,1,1);
	dim3 dimGrid1D(numYPerChunk*numTTot*2, 1, 1);

	int32_t CHIDX = (ChunkIndex + numChunks) % numChunks;

	cudaMemcpy(gChunkIndex, &CHIDX, sizeof(int32_t), cudaMemcpyHostToDevice);

	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Memcpy failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	cudaMemcpy(GPU_FLIMRawData_1Chunk, RawData_Block, sizeof(uint16_t) * numX * numYPerChunk * numTTot, cudaMemcpyHostToDevice);

	if (FILESTUFFEH)
		fprintf(LogFile, "Raw Data Memcpy for chunk %d (%d): %s\n", ChunkIndex, CHIDX, cudaGetErrorString(cudaPeekAtLastError()));

	cudaDeviceSynchronize();

	CountPeaks << <dimGrid1D, dimBlock1D >> > (GPU_FLIMRawData_1Chunk, GPU_numXxnumYxnumTxnumPulses_Count_Container, gNumxidMax);

	cudaDeviceSynchronize(); 

	AverageCountsToAPulse << <dimGrid3D, dimBlock3D >> > (GPU_numXxnumYxnumTxnumPulses_Count_Container, GPU_numXxnumYxnumT_Container, gnumX, gnumYPerChunk,
		gnumTTot, gnumT, gDivFactor, gChunkIndex);

	cudaDeviceSynchronize();
	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Generating Fall-off curve failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	if (FILESTUFFEH) fclose(LogFile);
	return cudaPeekAtLastError();
}

// Shift, bin, and calculate intensity and lifetime for one frame
DLLEXPORT cudaError_t FastFLIMGPU_DoFastFLIM(float* MPMImage, AvgDataType* OutputHistogram, float* OutputFLIM, uint16_t* Output_G, uint16_t* Output_S,
	int32_t numX, int32_t numY, int32_t numT, float *Thresh, uint8_t SaveEh_MPM, 
	char* SaveName_MPM, uint8_t SaveEh_LifeTime, char* SaveName_LifeTime)
{
	FILE* LogFile;
	if (FILESTUFFEH) fopen_s(&LogFile, FLIMLOGFILENAME, "a");

	//2D grid
	dim3 dimBlock2D(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid2D((numX) / dimBlock2D.x, (numY) / dimBlock2D.y);

	//3D grid
	dim3 dimBlock3D(BLOCK_SIZE_4, BLOCK_SIZE_4, BLOCK_SIZE_T);
	dim3 dimGrid3D((numX) / dimBlock3D.x, (numY) / dimBlock3D.y, (numT) / dimBlock3D.z);

	cudaMemcpy(gThresh, Thresh, sizeof(float), cudaMemcpyHostToDevice);

	//Shift the histogram of each pixel so that the max value is at the 0th timepoint 
	MaxMinShift << <dimGrid2D, dimBlock2D >> > (GPU_numXxnumYxnumT_Container, GPU_numXxnumYxnumT_Container_Shifted, gnumX, gnumT, gnumXnumYnumT);
	cudaDeviceSynchronize();
	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Max and Min shifting failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	//Optional averaging filter to increase the signal per pixel
	FluorescenceDecayBin << <dimGrid3D, dimBlock3D >> > (GPU_numXxnumYxnumT_Container_Shifted, GPU_numXxnumYxnumT_Container, gnumX, gnumY, gnumT);
	cudaDeviceSynchronize();
	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Binning failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	//Calcilatte intensity, lifetime, g, and s
	DoingMPMAndFLIMKernel << <dimGrid2D, dimBlock2D >> > (GPU_numXxnumYxnumT_Container, GPU_numXxnumY_MPMImage_BeforeAv, gnumX, gnumT,
		gThresh, GPU_numXxnumY_G_AfterAv, GPU_numXxnumY_S_AfterAv, GPU_numXxnumY_MeanLifetime_AfterAv);
	cudaDeviceSynchronize();
	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Generating MPM and FLIM failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	//Copy intensity and histogram to host
	cudaMemcpy(MPMImage, GPU_numXxnumY_MPMImage_BeforeAv, sizeof(float) * numX * numY, cudaMemcpyDeviceToHost);
	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Memcpy2 failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	cudaMemcpy(OutputHistogram, GPU_numXxnumYxnumT_Container, sizeof(AvgDataType) * numX * numY * numT, cudaMemcpyDeviceToHost);
	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Memcpy2 failed for decay: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	//Save intensity and histogram
	if (SaveEh_MPM)
	{
		FILE* FID_MPM = fopen(SaveName_MPM, "wb");
		fwrite(MPMImage, sizeof(float), numX * numY, FID_MPM);
		fclose(FID_MPM);
	}

	//Copy lifetime to host
	cudaMemcpy(OutputFLIM, GPU_numXxnumY_MeanLifetime_AfterAv, sizeof(float) * numX * numY, cudaMemcpyDeviceToHost);
	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Copying Average Lifetime Image failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	//Save lifetime
	if (SaveEh_LifeTime)
	{
		FILE* FID_LifeTime = fopen(SaveName_LifeTime, "wb");
		fwrite(OutputFLIM, sizeof(float), numX * numY, FID_LifeTime);
		fclose(FID_LifeTime);
	}

	//Copy s and g to host
	cudaMemcpy(Output_G, GPU_numXxnumY_G_AfterAv, sizeof(uint16_t) * numX * numY, cudaMemcpyDeviceToHost);
	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Copying Average G Values failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	cudaMemcpy(Output_S, GPU_numXxnumY_S_AfterAv, sizeof(uint16_t) * numX * numY, cudaMemcpyDeviceToHost);
	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Copying Average S Values failed: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	if (FILESTUFFEH) fclose(LogFile);
	return cudaPeekAtLastError();

	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Something in the end: %s\n", cudaGetErrorString(cudaPeekAtLastError()));

	if (FILESTUFFEH) fclose(LogFile);
	return cudaPeekAtLastError();
}

//Free memory!
DLLEXPORT cudaError_t FastFLIMGPU_DestroyEverything()
{
	FILE* LogFile;
	if (FILESTUFFEH) fopen_s(&LogFile, FLIMLOGFILENAME, "a");
	if (FILESTUFFEH) fprintf(LogFile, "CLEARING MEMORY\n");
	cudaFree(gnumX);
	cudaFree(gnumY);
	cudaFree(gnumT);
	cudaFree(gnumTTot);
	cudaFree(gDivFactor);
	cudaFree(gThresh);
	cudaFree(gChunkIndex);
	cudaFree(gNumChunks);
	cudaFree(gnumYPerChunk);
	cudaFree(GPU_numXxnumY_S_AfterAv);
	cudaFree(GPU_numXxnumY_G_AfterAv);
	cudaFree(GPU_Thresh_Array);
	cudaFree(GPU_FLIMRawData_1Chunk);
	cudaFree(GPU_numXxnumYxnumT_Container);
	cudaFree(GPU_numXxnumYxnumT_Container_Shifted);
	cudaFree(GPU_numXxnumY_MeanLifetime_AfterAv);
	cudaFree(GPU_numXxnumY_MPMImage_BeforeAv);
	cudaFree(gnumXnumYnumT);
	cudaFree(GPU_numXxnumYxnumTxnumPulses_Count_Container);
	cudaFree(gNumxidMax);


	if (FILESTUFFEH && (cudaPeekAtLastError() != cudaSuccess))
		fprintf(LogFile, "Clearing memory failed : %s\n", cudaGetErrorString(cudaPeekAtLastError()));
	if (FILESTUFFEH) fclose(LogFile);
	return cudaPeekAtLastError();
}
