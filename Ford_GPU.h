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
              int *devRowPtr, int *devDist, int* devRange, bool* devChanged){
    // allocate memory for the graph on device.
    const int M = A.num_edges();
    const int N = A.num_nodes();
    const int sizeDev  = M * sizeof(int));
    const int sizeDist = N * sizeof(int));

    cudaMalloc( (void**)&devVal, sizeDev );
    cudaMalloc( (void**)&devColInd, sizeDev );
    cudaMalloc( (void**)&devRowPtr, sizeDev + sizeof(int));
    cudaMalloc( (void**)&devDist, sizeDist );
    //    cudaMalloc( (void**)&devRange, 1 );
    //    cudaMalloc( (void**)&devChanged, 1 );


    // copy graph from host to device.
    cudaMemcpy( devVal,     A._val      , size    , cudaMemcpyHostToDevice);
    cudaMemcpy( devColInd,  A._col_ind  , sizeEdge    , cudaMemcpyHostToDevice);
    cudaMemcpy( devRowPtr,  A._row_ptr  , sizeEdge + sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( devDist,    dist        , sizeNode       , cudaMemcpyHostToDevice);
    //    cudaMemcpy( devRange,   RANGE       , 1       , cudaMemcpyHostToDevice);
    //    cudaMemcpy( devChanged, changed     , 1       , cudaMemcpyHostToDevice);
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

    //Device memory container
    int *devVal = 0, *devColInd = 0, *devRowPtr = 0 , *devDist = 0;
    int *devRange = 0;
    bool *devChanged = 0;

    init_GPU(A, dist, devVal, devColInd, devRowPtr, devDist, devRange, devChanged );

    const int N = A.num_nodes();
    const int M = A.num_edges();
    int RANGE = ceil((float) N / NUM_THREADS);

    int *arr = new int[M];
    cudaMemcpy( devColInd, arr  , M * sizeof(int), cudaMemcpyDeviceToHost);
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
    cudaMemcpy( dist, devDist , N, cudaMemcpyDeviceToHost);

    //Free Cuda Memory
    free_GPU(devVal, devColInd, devRowPtr, devDist);
}
#endif // !_FORD_GPU_H
