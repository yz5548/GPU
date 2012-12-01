#ifndef _FORD_GPU_H
#define _FORD_GPU_H

#include <math.h>
#include "CRS.h"
using namespace std;

/**
 * Relax all edges surrounding node u
 * @param u: source node to relax around
 */
__global__ void sssp(int *devVal, int *devColInd, int *devRowPtr, int *devDist,
          int *RANGE, bool *changed) {
    const int threadID = threadIdx.x;
    const int LEFT  = (*RANGE) * threadID;
    const int RIGHT = (*RANGE) *(threadID + 1);
    int cost, v, num_edge, begin, end, weight;

    for (int u = LEFT; u < RIGHT; ++u) {
        begin = devRowPtr[ u ];
        end  = devRowPtr[u + 1];
        num_edge =  end - begin ;
         for (int e = 0; e < num_edge; ++e) {
//             v = A.vertex(u, e);
             v =  devColInd[ begin + e];
             weith = devVal[ begin + e];
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
void init_GPU(CRS& A, int dist[], int *devVal, int *devColInd, int *devRowPtr, int *devDist){
    // allocate memory for the graph on device.
    int size = A.num_edges();
    int N = A.num_nodes();
    cudaMalloc( (void**)&devVal, size );
    cudaMalloc( (void**)&devColInd, size );
    cudaMalloc( (void**)&devRowPtr, size + 1);
    cudaMalloc( (void**)&devDist, N );
    // copy graph from host to device.
    cudaMemcpy( devVal,     A._val      , size, cudaMemcpyHostToDevice);
    cudaMemcpy( devColInd,  A._col_ind  , size, cudaMemcpyHostToDevice);
    cudaMemcpy( devRowPtr,  A._row_ptr  , size + 1, cudaMemcpyHostToDevice);
    cudaMemcpy( devRowPtr,  dist  , N, cudaMemcpyHostToDevice);
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
    int *devVal, *devColInd, *devRowPtr, *devDist;

    init_GPU(A, dist, devVal, devColInd, devRowPtr, devDist );

    const int N = A.num_nodes();
    int RANGE = ceil((float) N / (float) NUM_THREADS);
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
