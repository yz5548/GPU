#ifndef _FORD_GPU_H
#define _FORD_GPU_H

#include <cuda.h>
#include "CRS.h"
#include "Distance.h"

using namespace std;

#define devVal(x, y) devVal[ (y) * (N) + (x)]
#define devColInd(x, y) devColInd[ (y) * (N) + (x)]
#define SOURCE 1
#define NUM_ARGS 3

/**
 * Relax all edges surrounding node u
 * @param u: source node to relax around
 */
__global__ void sssp(int *devVal, int *devColInd, int *devDist,
          int *devArgs) {
    const int threadID = blockIdx.x * blockDim.x +  threadIdx.x;

    const int TOTAL_THREADS = devArgs[0];
    __shared__ int* changed ;
    const int N = devArgs[2];
    changed = &devArgs[1];
 
    int cost, v, num_edge, weight;
    for (int u = threadID; u <= N; u+=TOTAL_THREADS) {
        num_edge =  devColInd(u, 0);
        for (int e = 1; e <= num_edge; ++e) {
             v      = devColInd(u , e);
             weight = devVal(u, e);
             //Crictical computation and decision
             cost = devDist[u] + weight;
             if (cost < devDist[v]){
	         atomicMin(&devDist[v], cost);
                 *changed  = 1;
	     }
        }
	__syncthreads();
    }
}
/**
 * Allocate memory for GPU
 */
void init_GPU(CRS& A, int dist[], int args[],
              int *devVal, int *devColInd,
              int *devDist, int *devArgs){
    const int N = A.num_nodes();
    const int sizeByte = A.sizeByte();
    const int sizeDist = N * sizeof(int);
    const int sizeArgs = NUM_ARGS * sizeof(int);

    // copy graph from host to device.    

    cudaMemcpyAsync( devVal   ,  A._val      , sizeByte  , cudaMemcpyHostToDevice);
    cudaMemcpyAsync( devColInd,  A._col_ind  , sizeByte  , cudaMemcpyHostToDevice);
    cudaMemcpyAsync( devDist  ,  dist        , sizeDist  , cudaMemcpyHostToDevice);
    cudaMemcpy( devArgs  ,  args        , sizeArgs  , cudaMemcpyHostToDevice);

}

void free_GPU(int *devVal, int *devColInd, int *devDist, int *devArgs){
    cudaFree(devVal);
    cudaFree(devColInd);
    cudaFree(devDist);
    cudaFree(devArgs);
}
/**
 * Parallel Ford Bellman
 * @A: the graph
 */
void Ford_GPU(CRS& A, int dist[], const int NUM_BLOCKS,
              int NUM_THREADS) {

    const int N = A.num_nodes();
    const int sizeByte = A.sizeByte();
    const int sizeDist = N * sizeof(int);
    const int sizeArgs = NUM_ARGS * sizeof(int);
    //Calculate Range and init the argument lists
    const int TOTAL_THREADS = min( NUM_THREADS * NUM_BLOCKS, N);
    int args[] = {TOTAL_THREADS, 0, N};

    //Device memory container
    int *devVal, *devColInd, *devDist, *devArgs;

    // allocate memory for the graph on device

    cudaMalloc( (void**)&devVal   , sizeByte );
    cudaMalloc( (void**)&devColInd, sizeByte );
    cudaMalloc( (void**)&devDist,   sizeDist );
    cudaMalloc( (void**)&devArgs,   sizeArgs );

    init_GPU(A, dist, args, devVal, devColInd, devDist, devArgs);

    /**
     * Running the Program in multiple Threads.
     */

    int& changed = args[1];

    do {
        changed = 0;
        cudaMemcpy( devArgs,  args, sizeArgs  , cudaMemcpyHostToDevice);
        sssp<<<NUM_BLOCKS, NUM_THREADS, sizeArgs>>>(devVal, devColInd, devDist, devArgs);
        cudaMemcpy( args,  devArgs, sizeArgs , cudaMemcpyDeviceToHost);
    } while (changed);

    //Copy back data to dist
    cudaMemcpy( dist, devDist , sizeDist, cudaMemcpyDeviceToHost);

    //Free Cuda Memory
    free_GPU(devVal, devColInd, devDist, devArgs);
}
#endif // !_FORD_GPU_H
