#ifndef _FORD_GPU_H
#define _FORD_GPU_H

#include <math.h>
#include "CRS.h"
using namespace std;
/**
 * Relax all edges surrounding node u
 * @param u: source node to relax around
 */
//__global__
void sssp(CRS *graph, int* dist, const int* RANGE, bool* changed) {
    int threadID = 0; //threadIdx.x;
    CRS& A = *graph;
    const int LEFT  = (*RANGE) * threadID;
    const int RIGHT = (*RANGE) *(threadID + 1);
    int cost, v, num_edge;
    for (int u = LEFT; u < RIGHT; ++u) {
         num_edge = A.num_edges(u);
         for (int e = 0; e < num_edge; ++e) {
             v = A.vertex(u, e);
             //Crictical computation and decision
             cost = dist[u] + A(u, e);
             if (cost < dist[v]) {
                 *changed = true;
                 dist[v] = cost;
             }
         }
     }
}
/**
 * Parallel Ford Bellman
 * @A: the graph
 */
void Ford_GPU(CRS& A, int dist[], const int NUM_BLOCKS,
              const int NUM_THREADS) {

    const int N = A.num_nodes();
    int RANGE = ceil((float) N / (float) NUM_THREADS);
    /**
     * Running the Program in multiple Threads.
     */
    bool changed;
    do {
        changed = false;
//        sssp<<<NUM_BLOCKS,NUM_THREADS>>>(&A, dist, &RANGE, &changed);
        sssp(&A, dist, &RANGE, &changed);
    } while (changed);
}
#endif // !_FORD_GPU_H
