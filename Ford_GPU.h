#ifndef _FORD_GPU_H
#define _FORD_GPU_H

#include <cuda.h>
#include "CRS.h"
#include "Distance.h"

#define devVal(x, y) devVal[ (y) * (NUM_NODES) + (x)]
#define devColInd(x, y) devColInd[ (y) * (NUM_NODES) + (x)]

using namespace std;

/**
 * Relax all edges surrounding node u
 * @param u: source node to relax around
 */
__global__ void sssp(int *devVal, int *devColInd, int *devDist,
          int *devArgs) {
    const int threadID = blockIdx.x * blockDim.x +  threadIdx.x;

    const int TOTAL_THREADS = devArgs[0];
    int& changed    = devArgs[1];
    const int NUM_NODES     = devArgs[2];
    int cost, v, num_edge, weight;
    for (int u = threadID; u < NUM_NODES; u+= TOTAL_THREADS) {
        num_edge =  devColInd(u, 0);
        for (int e = 1; e <= num_edge; ++e) {
             v      = devColInd(u , e);
             weight = devVal(u, e);
             //Crictical computation and decision
             cost = devDist[u] + weight;
             if (devDist[v]> cost){
                 devDist[v] = cost;
                 changed = true;
             }
         }
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
    const int sizeArgs = 3 * sizeof(int);
    // copy graph from host to device.    

    cudaMemcpy( devVal   ,  A._val      , sizeByte  , cudaMemcpyHostToDevice);
    cudaMemcpy( devColInd,  A._col_ind  , sizeByte  , cudaMemcpyHostToDevice);
    cudaMemcpy( devDist  ,  dist        , sizeDist  , cudaMemcpyHostToDevice);
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
    const int sizeArgs = 3 * sizeof(int);

    //Calculate Range and init the argument lists
    const int TOTAL_THREADS =  min(NUM_THREADS * NUM_BLOCKS, N);
    NUM_THREADS = min(NUM_THREADS, N);
    int args[] = {TOTAL_THREADS, 0, N};

    //Device memory container
    int *devVal, *devColInd, *devArgs, *devDist;

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
	sssp<<<NUM_BLOCKS, NUM_THREADS>>>(devVal, devColInd, devDist, devArgs);
        cudaMemcpy( args,  devArgs, sizeArgs  , cudaMemcpyDeviceToHost);
    } while (changed);

    //Copy back data to dist
    cudaMemcpy( dist, devDist, sizeDist, cudaMemcpyDeviceToHost);

    //Free Cuda Memory
    free_GPU(devVal, devColInd, devDist, devArgs);
}
#endif // !_FORD_GPU_H
