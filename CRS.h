#ifndef _CRS_H
#define _CRS_H

#include <vector>
#include <limits.h>
#include <cassert>
#include <algorithm>

const int DEFAULT_SIZE = 100;
#define _val(x, y) _val[ (y) * (_NUM_NODES) + (x)]
#define _col_ind(x, y) _col_ind[ (y) * (_NUM_NODES) + (x)]

class CRS {
  public:
    int* _val;
    int* _col_ind;
    const int _NUM_NODES;
    const int _NUM_EDGES;
    const int _MAX_DEGREE;
  public:
    int num_nodes() const;
    CRS(int NUM_NODES, int NUM_EDGES, int MAX_DEGREE);
    ~CRS();
    void insert(int x, int y, int weight);
    int num_edges(int x);
    int vertex(int x, int index);
    int &operator()(int x, int y);
    int sizeByte();
    void print();
};

/**
 * Constructor:
 * Require number of nodes, and number of edges,
 * and max number of connection one node could possible have
 */
CRS::CRS(int NUM_NODES, int NUM_EDGES, int MAX_DEGREE):
    _NUM_NODES(NUM_NODES),
    _NUM_EDGES(NUM_EDGES),
    _MAX_DEGREE(MAX_DEGREE) {

    _val = new int[NUM_NODES*MAX_DEGREE];
    _col_ind = new int[NUM_NODES*MAX_DEGREE];
    for (int i = 0; i < NUM_NODES; ++i) {
        _val(i, 0) = 0;
        _col_ind(i, 0) = 0;
    }
}
/**
 * Destructor
 */
CRS::~CRS(){
    delete [] _val;
    delete [] _col_ind;
}

/**
 * @return: number of nodes;
 */
int CRS::num_nodes() const{
    return _NUM_NODES;
}

/**
 * Insert an edge to the CPR CRS
 */
void CRS::insert(int x, int y, int weight) {
    //First time insertion on a node
    int index = _col_ind(x, 0) + 1;
    _col_ind(x, index) = y;
    _val(x, index) = weight;

    ++_col_ind(x, 0);
    ++_val(x, 0);
}

/**
 * @return: the edge from node,
 * @param: u node
 * @param: index index of the arc on node u
 */
int& CRS::operator () (int x, int index){
    assert(index < _val(x, 0));
    return _val(x, index + 1);
}

/**
 * @return number of edges of node x
 */
int CRS::num_edges(int x){
    assert (_col_ind(x, 0) == _val(x, 0));
    return _col_ind(x, 0);
}

/**
 * @return: the vertex of node x, arc index
 */
int CRS::vertex(int x, int index){
    assert(index < _col_ind(x, 0));
    return _col_ind(x, index+1);
}

/**
 * Print matrix containers
 */
void CRS::print() {
    for (int i = 0; i < _NUM_NODES; ++i) {
        for (int j = 0; j < _col_ind(i, 0); ++j) {
            printf("%d %d: %d\n", i , vertex(i, j), _val(i, j+1));
        }
    }
}

int CRS::sizeByte(){
    return _NUM_NODES * _MAX_DEGREE * sizeof(int);
}
#endif
