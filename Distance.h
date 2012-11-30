/*
 * Distance.h
 *
 *  Created on: Nov 10, 2012
 *      Author: ttn14
 */

#ifndef DISTANCE_H_
#define DISTANCE_H_

#define MAX_VALUE 1000000000

void dist_init(int dist[], int SOURCE, const int N) {
    for (int i = 0; i < N; ++i) {
        dist[i] = MAX_VALUE ;
    }
    dist[SOURCE] = 0;
}

void dist_verify(int dist[], Graph& A, const int N) {
    for (int x = 0; x < N; ++x) {
        std::list<Edge> edges = A[x];
        std::list<Edge>::iterator iterator;
        for ( iterator = edges.begin(); iterator != edges.end(); ++iterator) {
            Edge node = *iterator;
            int y = node._vertex;
            int weight = node._weight;
            assert(dist[x] + weight >= dist[y]);
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
