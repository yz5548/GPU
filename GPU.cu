/**
 * To run the program:
 *     g++ -lcppunit -ldl -Wall TestSSSP.c++ -o TestSSSP.app
 */
// --------
// includes
// --------
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
//#include <thrust/device_vector.h>

#include "Graph.h"
#include "Distance.h"
#include "Ford_GPU.h"
#include "CRS.h"

#ifndef MAX_DEG
#define MAX_DEG 100
#endif

using namespace std;

/**
 * Function Prototype
 */
void read_graph(CRS& A);
void read_graph_dimension(int& NUM_NODES, int& NUM_EDGES);
//void init_gpu(thrust::device_vector<int>& val, thrust::device_vector<int>& col_ind, thrust::device_vector<int>& row_ptr);
// ----
// main
// ----

int main(int argc, char** argv) {
    int NUM_NODES, NUM_EDGES;

    /**
     * Reading Graph Data
     */
    read_graph_dimension(NUM_NODES, NUM_EDGES);
    CRS A(NUM_NODES, NUM_EDGES, MAX_DEG);
    read_graph(A);

    const int NUM_BLOCKS  = atoi(argv[1]);
    const int NUM_THREADS = atoi(argv[2]);

    //Initialization
    int* dist = new int[NUM_NODES];
    dist_init(dist, SOURCE, NUM_NODES);

    Ford_GPU(A, dist, NUM_BLOCKS, NUM_THREADS);
    dist_verify(dist, A, NUM_NODES);

    delete dist;

    return 0;
}

/**
 * Read Graph Dimension
 * @return: graph number of nodes
 */

void read_graph_dimension(int& NUM_NODES, int& NUM_EDGES) {
    char line_type;
    char graph_type[5];
    char line[256];
    NUM_NODES = -1;
    NUM_EDGES = -1;

    while (NUM_NODES == -1 && cin >> line_type) {
        if (line_type == 'c') {
            cin.getline(line, 256, '\n');
        } else if (line_type == 'p') {
            cin >> graph_type;
            cin >> NUM_NODES;  // Number of nodes
            cin >> NUM_EDGES;  // Number of edges
        }
    }
    // Graph starts from Node 1
    NUM_NODES++;
    NUM_EDGES++;
}
/**
 * Read the input files.
 * Put data in graph's storage
 */
void read_graph(CRS& A) {
    char line_type;
    char line[256];
    int x, y, v;
    int weight;

    const int N = A.num_nodes();
    Graph B( N , A._NUM_EDGES);
    while (cin >> line_type) {
        if (line_type == 'c') {
            cin.getline(line, 256, '\n');
        } else if (line_type == 'a') {
            cin >> x >> y >> weight;
            B.insert(x, y, weight);
        } else
            break;
    }
    
    for (int u = 1; u < N; ++u) {
        list<Edge>& edges = B[u];
        list<Edge>::iterator iterator;
        for (iterator = edges.begin(); iterator != edges.end(); ++iterator) {
            Edge edge = *iterator;
            v = edge._vertex;
            A.insert( u, v, edge._weight);
        }
    }    
    
}

