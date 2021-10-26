//Program for performing GPU-accelerated single-photon peak event detection for FLIM
//By Janet Sorrells and Rishee Iyer, 2021
//Contact: janetes2@illinois.edu

//Include header files
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <iostream>
#include <stddef.h>
#include <memoryapi.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cassert>
#include <time.h>

//Define fixed variables
#define DLLEXPORT extern "C" __declspec(dllexport)
#define BLOCK_SIZE 4
#define BLOCK_SIZE_4 4
#define BLOCK_SIZE_4A 4
#define BLOCK_SIZE_T 8
#define FILESTUFFEH 0
#define BINSTEP 1 //BINSTEP is the number of adjacent pixels used for imaging, BINSTEP = 1 --> 3x3 binning
#define DECAYLENGTH 37 //Often the last few data points in the decay start going to higher values, 
//this way we ignore them and only look at the first 37 points in a pixel's histogram to calculate the lifetime
#define PI 3.14159265f
#define TAXIS_COEFF 2.0f*PI*80E6f/(3.2E9f)
#define LT_COEFF (1E12f/(2.0f*PI*80E6f))
#define FLIMLOGFILENAME "C:\\FLIM Acquisiton Software GPU Version\\FLIMPostProcLogFile_v107.txt" //Set this where you want the log file to go
#define PEAKTHRESH 26800 //Single-photon peak threshold, determine this from a histogram of peak heights, see publication for more info

//GPU pointers and variables
extern uint16_t * GPU_FLIMRawData_1Chunk;
extern float * GPU_256x256_MeanLifetime_AfterAv, * GPU_256x256_MPMImage_BeforeAv;
extern uint16_t* GPU_256x256_S_AfterAv, * GPU_256x256_G_AfterAv;
extern uint8_t* GPU_Thresh_Array;
extern float* gThresh;
extern int32_t* gnumX, * gnumY, * gnumTTot, * gnumT, * gDivFactor, * gNumChunks, *gChunkIndex, *gnumYPerChunk, *gnumXnumYnumT, *gNumxidMax;

typedef uint8_t CountDataType;
typedef uint16_t AvgDataType;

extern CountDataType* GPU_numXxnumYxnumTxnumPulses_Count_Container;
extern AvgDataType* GPU_numXxnumYxnumT_Container, * GPU_numXxnumYxnumT_Container_Shifted;

/*GPU Kernels below!!*/


/* Average counts to a "pulse" (create a histogram of photon counts vs. time where the time corresponds to the laser period)
InData: photon counts numX x numYPerChunk x numTtot
OutData: photon counts numX x numYPerChunk x numT 
*/
__global__ void AverageCountsToAPulse(CountDataType* InData, AvgDataType* OutData, int32_t* numX, int32_t* numYPerChunks, int32_t* numTTot,
	int32_t* numT, int32_t* DivFactor, int32_t* ChunkIndex)
{
	long xid = blockIdx.x * blockDim.x + threadIdx.x;
	long yid = blockIdx.y * blockDim.y + threadIdx.y;
	long zid = blockIdx.z * blockDim.z + threadIdx.z;
	size_t tidx;
	//size_t outidx = (size_t)zid + (size_t)xid * (size_t)*numT + (size_t)yid * (size_t)*numT * (size_t)*numX;
	size_t inidx = (size_t)zid + (size_t)xid * (size_t)*numTTot + (size_t)yid * (size_t)*numTTot * (size_t)*numX;
	//size_t ChunkSkip = (size_t)*numX * (size_t)*numYPerChunks * (size_t)*numT * (size_t)*ChunkIndex;

	// add outidx 8/31/21


	AvgDataType tempOut = 0;
	for (tidx = 0; tidx < *DivFactor; tidx++)
	{
		// tempOut += (AvgDataType)InData[inidx + tidx * (size_t)*numT]; //8/31/21
		tempOut += (AvgDataType)InData[inidx + (tidx * (size_t)*numT)];
	}
	OutData[(size_t)*numX * (size_t)*numYPerChunks * (size_t)*numT * (size_t)*ChunkIndex + (size_t)zid + (size_t)xid * (size_t)*numT + (size_t)yid * (size_t)*numT * (size_t)*numX] = tempOut;
}

/*Count photons - Single-photon PEak Event Detection (SPEED)!!!!
InData: PMT output amplified and digitized (12-bit), numX x numYperChunk x numTtot
OutData: 1 for photon count, 0 for no photon count, numX x numYperChunk x numTtot
PMT outputs are negative, so really this is inverted peak detection. 
Peaks are less than the peak threshold, and less than the two neighboring digitized points. 
*/
__global__ void CountPeaks(uint16_t* InData, CountDataType* OutData, int32_t* xidMax)
{
	int32_t xid = blockIdx.x * blockDim.x + threadIdx.x;

	uint16_t pastPoint, presentPoint, futurePoint;

	if (xid < *xidMax)
	{
		pastPoint = InData[xid];
		presentPoint = InData[xid + 1];
		futurePoint = InData[xid + 2];
	}
	else
	{
		pastPoint = 0U; 
		presentPoint = 0U;
		futurePoint = 0U;
	}

	if ((presentPoint <= PEAKTHRESH) && (presentPoint <= pastPoint) && (presentPoint <= futurePoint)) 
	{
		OutData[xid] = 1;
	}
	else
	{
		OutData[xid] = 0;
	}
	
}

/*Maximum Shifting: circularly shift each pixel's histogram so that the maximum value is at the first timepoint
*/
__global__ void MaxMinShift(AvgDataType* InData, AvgDataType* OutData, int32_t* numX, int32_t* numT, int32_t* numXnumYnumT)
{
	long xid = blockIdx.x * blockDim.x + threadIdx.x;
	long yid = blockIdx.y * blockDim.y + threadIdx.y;
	long inidx_off = xid * *numT + yid * *numT * *numX;
	long tid = 0;
	long MaxIdx = 0;
	long tid2 = 0;
	AvgDataType MaxVal = InData[inidx_off];

	if ((inidx_off > *numT) && (inidx_off < (*numXnumYnumT - *numT)))
	{
		for (tid = 0; tid < (*numT); tid++)
		{
			if ((InData[inidx_off + tid] + InData[inidx_off + tid - *numT] + InData[inidx_off + tid + *numT]) > MaxVal)
			{
				MaxIdx = tid;
				MaxVal = (InData[inidx_off + tid] + InData[inidx_off + tid - *numT] + InData[inidx_off + tid + *numT]); //revised 3/19/21
			}

		}
	}

	for (tid2 = 0; tid2 < *numT; tid2++)
	{
		OutData[inidx_off + tid2] = (InData[(inidx_off + tid2 + MaxIdx) % *numXnumYnumT]);
	}
}



/*Binning
*/
__global__ void FluorescenceDecayBin(AvgDataType* InData, AvgDataType* OutData, int32_t* numX, int32_t* numY, int32_t* numT)
{
	long xid = blockIdx.x * blockDim.x + threadIdx.x;
	long yid = blockIdx.y * blockDim.y + threadIdx.y;
	long tid = blockIdx.z * blockDim.z + threadIdx.z;
	//long outidx = xid * *numT + yid * *numT * *numX + tid;
	long xidx, yidx;
	AvgDataType outTemp = 0;

	for (xidx = -BINSTEP; xidx <= BINSTEP; xidx++)
	{
		for (yidx = -BINSTEP; yidx <= BINSTEP; yidx++)
		{
			outTemp += InData[((xid + xidx + *numX) % *numX) * *numT + ((yid + yidx + *numY) % *numY) * *numT * *numX + tid]; //exclude edge pixels
		}
	}
	OutData[xid * *numT + yid * *numT * *numX + tid] = outTemp;
}


/*Calculate multiphoton microscopy intensity and lifetime, and phasor components g and s
*/
__global__ void DoingMPMAndFLIMKernel(AvgDataType* InData, float* MPMImage, int32_t* numX, int32_t* numT, float* Thresh, uint16_t* GAv, uint16_t* SAv,
	float* Meanlifetime)
{
	long xid = blockIdx.x * blockDim.x + threadIdx.x;
	long yid = blockIdx.y * blockDim.y + threadIdx.y;
	long tidx;
	long outidx = xid + yid * *numX;
	long inidx = xid * *numT + yid * *numT * *numX;
	float G, S;

	MPMImage[outidx] = 0.0f;
	for (tidx = 0; tidx < *numT; tidx++)
	{
		MPMImage[outidx] += (InData[inidx + tidx]);
	}
	if (MPMImage[outidx] >= *Thresh)
	{
		float SumSine = 0.0f, SumCosine = 0.0f;
		AvgDataType SumSum = 0;
		for (tidx = 0; tidx < DECAYLENGTH; tidx++)
		{
			SumSine += ((float)InData[inidx + tidx] * sinf(TAXIS_COEFF * (tidx)));
			SumCosine += ((float)InData[inidx + tidx] * cosf(TAXIS_COEFF * (tidx)));
			SumSum += (InData[inidx + tidx]);
		}

		G = SumCosine / ((float)SumSum + 1E-15f);
		S = SumSine / ((float)SumSum + 1E-15f);
	}
	else
	{
		G = -1.0f;
		S = 0.0f;
	}
	if (S > 0.0f && G > 0.0f)
	{
		Meanlifetime[outidx] = (LT_COEFF * (S / G));
		SAv[outidx] = (uint16_t)(S * 10000.0f);
		GAv[outidx] = (uint16_t)(G * 10000.0f);
	}
	else
	{
		Meanlifetime[outidx] = 0.0f;
		SAv[outidx] = 0;
		GAv[outidx] = 0;
	}
}
