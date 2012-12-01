#ifndef _CRS_H
#define _CRS_H

#include <vector>
#include <limits.h>
#include <cassert>
#include <algorithm>


const int DEFAULT_SIZE = 100;

class CRS {
  private:
    int* _val;
    int* _col_ind;
    int* _row_ptr;
    const int _NUM_NODES;
    const int _NUM_EDGES;

  public:
    int num_nodes() const;
    CRS(int NUM_NODES, int NUM_EDGES);
    ~CRS();
    void insert(int x, int y, double weight);
    int num_edges(int x);
    int num_edges() const;
    int vertex(int x, int index);
    int &operator()(int x, int y);
    void print();
};

/**
 * Constructor:
 * Require number of nodes, and number of edges
 */
CRS::CRS(int NUM_NODES, int NUM_EDGES)
        : _val( new int[NUM_EDGES]),
          _col_ind( new int[NUM_EDGES]),
          // it needs an extra +1 to know where to stop
          _row_ptr( new int[NUM_NODES + 1]),

          _NUM_NODES(NUM_NODES),
          _NUM_EDGES(NUM_EDGES) {
    _row_ptr[1] = 1;
}

/**
 * Destructor
 */
CRS::~CRS(){
    delete _val;
    delete _col_ind;
    delete _row_ptr;
}

/**
 * @return: number of nodes;
 */
int CRS::num_nodes() const{
    return _NUM_NODES;
}
int CRS::num_edges() const{
    return _NUM_NODES;
}

/**
 * Insert an edge to the CPR CRS
 */
void CRS::insert(int x, int y, double weight) {
    //First time insertion on a node
    if (_row_ptr[x + 1] == 0) {
        _row_ptr[x + 1] = _row_ptr[x];
    }

    int index = _row_ptr[x + 1];
    _row_ptr[x + 1]++;
    _col_ind[index] = y;
    _val[index] = weight;
}

/**
 * @return: the edge from node,
 * @param: u node
 * @param: index index of the arc on node u
 */
int& CRS::operator () (int x, int index){
    int begin, end;
    begin = _row_ptr[ x ];
    end  = _row_ptr[ x + 1 ];
    assert (begin + index < end);
    return _val[ begin + index];
}

/**
 * @return number of edges of node x
 */
int CRS::num_edges(int x){
    int num = std::max(_row_ptr[x + 1] - _row_ptr[ x ], 0);
    return num;
}

/**
 * @return: the vertex of node x, arc index
 */
int CRS::vertex(int x, int index){
    int begin, end;
    begin = _row_ptr[ x ];
    end  = _row_ptr[ x + 1 ];
    assert (begin + index < end);
    return _col_ind[ begin + index];
}

/**
 * Print matrix containers
 */
void CRS::print() {
    for (unsigned int i = 1; i < _NUM_NODES; ++i) {
        std::cout << _val[i] << " ";
    }
    std::cout << std::endl;
    for (unsigned int i = 1; i < _NUM_NODES; ++i) {
        std::cout << _col_ind[i] << " ";
    }
    std::cout << std::endl;
    for (unsigned int i = 1; i < _NUM_NODES; ++i) {
        std::cout << _row_ptr[i] << " ";
    }
    std::cout << std::endl;
}
#endif
