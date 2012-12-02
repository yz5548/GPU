#ifndef _FORD_GPU_H
#define _FORD_GPU_H

#include <cuda.h>
#include "CRS.h"
#include "Distance.h"

using namespace std;

/**
 * Relax all edges surrounding node u
 * @param u: source node to relax around
 */
__global__ void sssp(int *devVal, int *devColInd, int *devRowPtr, int *devDist,
          int *devArgs) {
    const int threadID = threadIdx.x;

    const int RANGE = devArgs[0];
    int& changed    = devArgs[1];

    const int LEFT  = RANGE * threadID;
    const int RIGHT = RANGE *(threadID + 1);
    int cost, v, num_edge, begin, end, weight;
    for (int u = LEFT; u < RIGHT; ++u) {
        begin = devRowPtr[u];
        end   = devRowPtr[u + 1];
	
        num_edge =  end - begin ;
        for (int e = 0; e < num_edge; ++e) {
             v      = devColInd[ begin + e];
             weight = devVal[begin + e];
             //Crictical computation and decision
             cost = devDist[u] + weight;

             if (cost < devDist[v]) {
                 changed = true;
                 devDist[v] = cost;
             }

         }
	
     }

}

/**
 * Allocate memory for GPU
 */
void init_GPU(CRS& A, int dist[], int args[],
              int *devVal, int *devColInd, 
              int *devRowPtr, int *devDist, int *devArgs){
    const int M = A.num_edges();
    const int N = A.num_nodes();
    const int  sizeVal    = M * sizeof(int); 
    const int& sizeColInd = sizeVal;
    const int sizeRowPtr  = (N + 1) * sizeof(int);
    const int sizeDist    = N * sizeof(int);
    const int sizeArgs    = 2 * sizeof(int);

    // copy graph from host to device.    

    cudaMemcpy( devVal   ,  A._val      , sizeVal   , cudaMemcpyHostToDevice);
    cudaMemcpy( devColInd,  A._col_ind  , sizeColInd, cudaMemcpyHostToDevice);
    cudaMemcpy( devRowPtr,  A._row_ptr  , sizeRowPtr, cudaMemcpyHostToDevice);
    cudaMemcpy( devDist  ,  dist        , sizeDist  , cudaMemcpyHostToDevice);
    cudaMemcpy( devArgs  ,  args        , sizeArgs  , cudaMemcpyHostToDevice);

}

void free_GPU(int *devVal, int *devColInd, int *devRowPtr, int *devDist, int *devArgs){
    cudaFree(devVal);
    cudaFree(devColInd);
    cudaFree(devRowPtr);
    cudaFree(devDist);
    cudaFree(devArgs);
}
/**
 * Parallel Ford Bellman
 * @A: the graph
 */
void Ford_GPU(CRS& A, int dist[], const int NUM_BLOCKS,
              const int NUM_THREADS) {
    

    const int N = A.num_nodes();
    const int M = A.num_edges();
    const int  sizeVal  = M * sizeof(int); 
    const int& sizeColInd = sizeVal;
    const int sizeRowPtr = (N + 1) * sizeof(int);
    const int sizeDist = N * sizeof(int);
    const int sizeArgs = 2 * sizeof(int);

    //Calculate Range and init the argument lists
    const int RANGE = ceil(N / (float)NUM_THREADS);
    int args[] = {RANGE, 0};

    //Device memory container
    int *devVal, *devColInd, *devRowPtr, *devDist, *devArgs;

    // allocate memory for the graph on device
    cudaMalloc( (void**)&devVal   , sizeVal );
    cudaMalloc( (void**)&devColInd, sizeColInd );
    cudaMalloc( (void**)&devRowPtr, sizeRowPtr);
    cudaMalloc( (void**)&devDist,   sizeDist );
    cudaMalloc( (void**)&devArgs,   sizeArgs );


    init_GPU(A, dist, args, devVal, devColInd, devRowPtr, devDist, devArgs);


    /**
     * Running the Program in multiple Threads.
     */


    int* arr1 = new int [N+1];
    int& changed = args[1];

    do {        
        changed = 0;
        cudaMemcpy( devArgs,  args, sizeArgs  , cudaMemcpyHostToDevice);
        sssp<<<NUM_BLOCKS, NUM_THREADS>>>(devVal, devColInd, devRowPtr, devDist, devArgs);
        cudaMemcpy( args,  devArgs, sizeArgs  , cudaMemcpyDeviceToHost);
    } while (changed);
  
    free(arr1);  

    //Copy back data to dist
    cudaMemcpy( dist, devDist , sizeDist, cudaMemcpyDeviceToHost);

    //Free Cuda Memory
    free_GPU(devVal, devColInd, devRowPtr, devDist, devArgs);
}
#endif // !_FORD_GPU_H
