#ifndef _FORD_GPU_H
#define _FORD_GPU_H

#include <math.h>
using namespace std;
/**
 * Relax all edges surrounding node u
 * @param u: source node to relax around
 */
//__global__
void sssp(Graph& A, int dist[], const int& RANGE, bool& changed) {
    int threadID = threadIdx.x;
    const int LEFT = threadID * RANGE;
    const int RIGHT = (threadID + 1) * RANGE;
    int cost, v;
    for (int u = LEFT; u < RIGHT; ++u) {
        list<Edge>& edges = A[u];
        list<Edge>::iterator iterator;
        for (iterator = edges.begin(); iterator != edges.end(); ++iterator) {
            Edge edge = *iterator;
            v = edge._vertex;
            cost = dist[u] + edge._weight;
            if (cost < dist[v]) {
                changed = true;
                dist[v] = cost;
            }
        }
    }
}
/**
 * Parallel Ford Bellman
 * @A: the graph
 */
void Ford_GPU(Graph& A, int dist[], const int NUM_BLOCKS,
              const int NUM_THREADS) {

    const int N = A.num_nodes();
    int RANGE = ceil((float) N / (float) NUM_THREADS);
    /**
     * Running the Program in multiple Threads.
     */
    bool changed;
    do {
        changed = false;
//        sssp<<<NUM_BLOCKS,NUM_THREADS>>>(A, dist, RANGE, changed);
        sssp(A, dist, RANGE, changed);
    } while (changed);
}
#endif // !_FORD_GPU_H
