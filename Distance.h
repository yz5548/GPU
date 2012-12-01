/*
 * Distance.h
 *
 *  Created on: Nov 10, 2012
 *      Author: ttn14
 */

#ifndef DISTANCE_H_
#define DISTANCE_H_

#define MAX_VALUE 1000000000
#include "CRS.h"
void dist_init(int dist[], int SOURCE, const int N) {
    for (int i = 0; i < N; ++i) {
        dist[i] = MAX_VALUE ;
    }
    dist[SOURCE] = 0;
}

void dist_verify(int dist[], CRS& A, const int N) {
    for (int x = 0; x < N; ++x) {
        const int NUM_EDGES = A.num_edges(x);
        for (int edge_index = 0; edge_index < NUM_EDGES; ++edge_index) {
            int y = A.vertex(x, edge_index);
            assert(dist[x] + A(x, edge_index) >= dist[y]);
        }
    }
}

void dist_print(int dist[], const int N) {
    for (int i = 1; i < N; ++i){
        std::cout << i;
        if (dist[i] != MAX_VALUE) {
             std::cout << " " << dist[i] << std::endl;
        }
        else
            std::cout << " " << "INF" << std::endl;
    }
}

#endif /* DISTANCE_H_ */
