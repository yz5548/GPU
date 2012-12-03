#ifndef _FORD_GPU_H
#define _FORD_GPU_H

#include <cuda.h>

#include "CRS.h"
#include "Distance.h"

#define devVal(x, y) devVal[ (y) * (NUM_NODES) + (x)]
#define devColInd(x, y) devColInd[ (y) * (NUM_NODES) + (x)]
#define SOURCE 1

using namespace std;

void print_work(unsigned *work, int sizeWork){
    int* hostWork = new int[sizeWork];
    const int N = sizeWork / sizeof(int);
    cudaMemcpy( hostWork, work, sizeWork, cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; ++i)
        cout << hostWork[i] << " " ;    
    cout << endl;
}

/**
 * Relax all edges surrounding node u
 * @param u: source node to relax around
 */
__global__ void sssp(int *devVal, int *devColInd, int *devDist,
		     int *devArgs, unsigned* oldWork, unsigned* newWork) {

    const int threadID = blockIdx.x * blockDim.x +  threadIdx.x;

    const int TOTAL_THREADS = devArgs[0];
    int& changed    = devArgs[1];

    const int NUM_NODES     = devArgs[2];

    const int WORK_SIZE     = oldWork[0];

    int cost, u, v, num_edge, weight, wIndex;
    for (int i = threadID + 1; i <= WORK_SIZE; i+= TOTAL_THREADS) {
        u = oldWork[i];
        num_edge =  devColInd(u, 0);
        for (int e = 1; e <= num_edge; ++e) {
             v      = devColInd(u , e);
             weight = devVal(u, e);
             //Crictical computation and decision
             cost = devDist[u] + weight;
	     if (cost < devDist[v] ){
	         atomicMin(&devDist[v], cost);
		 changed = true;
		 atomicAdd(&newWork[0], 1);
		 wIndex = newWork[0];
		 atomicExch( &newWork[wIndex], v);
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

void free_GPU(int *devVal, int *devColInd, int *devDist, int *devArgs
	      , unsigned* oldWork, unsigned* newWork){
    cudaFree(devVal);
    cudaFree(devColInd);
    cudaFree(devDist);
    cudaFree(devArgs);
    cudaFree(oldWork);
    cudaFree(newWork);
}

void init_work(unsigned* work, int sizeWork, int init = 0){
    int* empty = new int[sizeWork];
    const int N = sizeWork / sizeof(int);
    for (int i = 0; i < N; ++i)
        empty[i] = 0;

    if (init){
       ++empty[0];
       empty[1] = SOURCE;
    }

    cudaMemcpy( work, empty, sizeWork, cudaMemcpyHostToDevice);
    free(empty);
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
    const int sizeWork = sizeDist;

    //Calculate Range and init the argument lists
    const int TOTAL_THREADS =  min(NUM_THREADS * NUM_BLOCKS, N);
    NUM_THREADS = min(NUM_THREADS, N);
    int args[] = {TOTAL_THREADS, 0, N};

    //Device memory container
    int *devVal, *devColInd, *devArgs, *devDist;
    unsigned *oldWork, *newWork;

    // allocate memory for the graph on device
    cudaMalloc( (void**)&devVal   , sizeByte );
    cudaMalloc( (void**)&devColInd, sizeByte );
    cudaMalloc( (void**)&devDist,   sizeDist );
    cudaMalloc( (void**)&devArgs,   sizeArgs );
    cudaMalloc( (void**)&oldWork,   sizeWork );
    cudaMalloc( (void**)&newWork,   sizeWork );

    init_GPU(A, dist, args, devVal, devColInd, devDist, devArgs);


    /**
     * Running the Program in multiple Threads.
     */

    int& changed = args[1];
    init_work( newWork, sizeWork, 1);

    do {
        changed = 0;
        
        cudaMemcpy( devArgs,  args, sizeArgs  , cudaMemcpyHostToDevice);
	cudaMemcpy( oldWork,  newWork, sizeWork  , cudaMemcpyDeviceToDevice);
	print_work( oldWork, sizeWork);
	init_work( newWork, sizeWork);

	sssp<<<NUM_BLOCKS, NUM_THREADS, sizeWork>>> (devVal, devColInd, devDist, devArgs, oldWork, newWork);

        cudaMemcpy( args, devArgs, sizeArgs, cudaMemcpyDeviceToHost);
    } while (changed);

    //Copy back data to dist
    cudaMemcpy( dist, devDist, sizeDist, cudaMemcpyDeviceToHost);

    //Free Cuda Memory
    free_GPU(devVal, devColInd, devDist, devArgs, oldWork, newWork);
}
#endif // !_FORD_GPU_H
