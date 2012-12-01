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
          int *RANGE, bool *changed) {
  const int threadID = threadIdx.x;
    const int LEFT  = 0;//(*RANGE) * threadID;
    const int RIGHT = 7;//(*RANGE) *(threadID + 1);
    int cost, v, num_edge, begin, end, weight;

    for (int u = LEFT; u < RIGHT; ++u) {
        begin = devRowPtr[ u ];
        end  = devRowPtr[u + 1];
        num_edge =  end - begin ;
         for (int e = 0; e < num_edge; ++e) {
             v =  devColInd[ begin + e];
             weight = devVal[ begin + e];
             //Crictical computation and decision
             cost = devDist[u] + weight;
             if (cost < devDist[v]) {
                 *changed = true;
                 devDist[v] = cost;
             }
         }
     }
}

/**
 * Allocate memory for GPU
 */
void init_GPU(CRS& A, int dist[], 
              int *devVal, int *devColInd, 
              int *devRowPtr, int *devDist){
    // allocate memory for the graph on device.
    const int M = A.num_edges();
    const int N = A.num_nodes();
    const int sizeA  = M * sizeof(int);
    const int sizeRowPtr = (M + 1) * sizeof(int)
    const int sizeDist = N * sizeof(int);

    cudaMalloc( (void**)&devVal   , sizeA );
    cudaMalloc( (void**)&devColInd, sizeA );
    cudaMalloc( (void**)&devRowPtr, sizeA + sizeof(int));
    cudaMalloc( (void**)&devDist, sizeDist );

    // copy graph from host to device.
    cudaMemcpy( devVal,     A._val      , sizeA     , cudaMemcpyHostToDevice);
    cudaMemcpy( devColInd,  A._col_ind  , sizeA     , cudaMemcpyHostToDevice);
    cudaMemcpy( devRowPtr,  A._row_ptr  , sizePtr   , cudaMemcpyHostToDevice);
    cudaMemcpy( devDist,    dist        , sizeDist  , cudaMemcpyHostToDevice);
}

void free_GPU(int *devVal, int *devColInd, int *devRowPtr, int *devDist){
    cudaFree(devVal);
    cudaFree(devColInd);
    cudaFree(devRowPtr);
    cudaFree(devDist);
}
/**
 * Parallel Ford Bellman
 * @A: the graph
 */
void Ford_GPU(CRS& A, int dist[], const int NUM_BLOCKS,
              const int NUM_THREADS) {
    int RANGE = ceil((float) N / NUM_THREADS);

    //Device memory container
    int *devVal = 0, *devColInd = 0, *devRowPtr = 0 , *devDist = 0;

    init_GPU(A, dist, devVal, devColInd, devRowPtr, devDist);

    const int N = A.num_nodes();
    const int M = A.num_edges();
    const int sizeA  = M * sizeof(int);
    const int sizeRowPtr = (M + 1) * sizeof(int)
    const int sizeDist = N * sizeof(int);

    int *arr = new int[M];
    cudaMemcpy( devVal, arr  , sizeA, cudaMemcpyDeviceToHost);
    dist_print(arr, M);
    delete arr;

    /**
     * Running the Program in multiple Threads.
     */
    bool changed;
    do {        
        changed = false;
        //TODO: copy Range and changed to GPU as well
        sssp<<<NUM_BLOCKS,NUM_THREADS>>>(devVal, devColInd, devRowPtr, devDist, &RANGE, &changed);
    } while (changed);
  
    //Copy back data to dist
    cudaMemcpy( dist, devDist , sizeDist, cudaMemcpyDeviceToHost);

    //Free Cuda Memory
    free_GPU(devVal, devColInd, devRowPtr, devDist);
}
#endif // !_FORD_GPU_H
