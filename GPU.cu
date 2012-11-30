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

#include "Graph.h"
#include "Distance.h"
#include "Ford_GPU.h"

using namespace std;

/**
 * Function Prototype
 */
void read_graph(Graph& A);
void read_graph_dimension(int& NUM_NODES, int& NUM_EDGES);

// ----
// main
// ----

int main(int argc, char** argv) {
    int NUM_NODES, NUM_EDGES;

    /**
     * Reading Graph Data
     */
    read_graph_dimension(NUM_NODES, NUM_EDGES);
    Graph A(NUM_NODES, NUM_EDGES);
    read_graph(A);


    const int SOURCE = 1;
    const int NUM_BLOCKS  = atoi(argv[1]);
    const int NUM_THREADS = atoi(argv[2]);

    //Initialization
    int* dist = new int[NUM_NODES];
    dist_init(dist, SOURCE, NUM_NODES);

    // allocate memory for the graph on device.
    int* devA;
    int size = 100;
    cudaMalloc((void**)&devA,size);


    // copy graph from host to device.

    Ford_GPU(A, dist, NUM_BLOCKS, NUM_THREADS);

    dist_print(dist, NUM_NODES);
    dist_verify(dist, A, NUM_NODES);

    delete dist;

//    int start = clock();
//    int stop = clock();
//    int elapsed_time = stop - start;
//    cout << "Execution Time: " << ((float) elapsed_time) / CLOCKS_PER_SEC << endl;
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
void read_graph(Graph& A) {
    char line_type;
    char line[256];
    int x, y;
    int weight;

    while (cin >> line_type) {
        if (line_type == 'c') {
            cin.getline(line, 256, '\n');
        } else if (line_type == 'a') {
            cin >> x >> y >> weight;
            A.insert(x, y, weight);
        } else
            break;
    }
}

