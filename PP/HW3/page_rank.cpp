
#include "page_rank.h"

#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include <utility>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

// pageRank --
//
// g:           graph to process (see common/graph.h)
// solution:    array of per-vertex vertex scores (length of array is num_nodes(g))
// damping:     page-rank algorithm's damping parameter
// convergence: page-rank algorithm's convergence threshold
//
void pageRank(Graph g, double *solution, double damping, double convergence)
{

  // initialize vertex weights to uniform probability. Double
  // precision scores are used to avoid underflow for large graphs

  int numNodes = num_nodes(g);
  double equal_prob = 1.0 / numNodes;
  double *score_old;
  score_old = new double[numNodes];
  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < numNodes; ++i)
    {
      solution[i] = equal_prob;
      score_old[i] = equal_prob;
    }
  }

  bool converged = false;
  while (!converged) {
    double no_outgoing = 0.0;
    double global_diff = 0.0;
    #pragma omp parallel 
    {
      #pragma omp for reduction (+:no_outgoing)
      for (int vi = 0 ; vi < numNodes ; vi++)
      {
        if (outgoing_size(g, vi) == 0)
        {
          no_outgoing += damping * score_old[vi] / numNodes;
        }
      }
      #pragma omp for
      for (int v=0;v<numNodes;v++)
      {
        const Vertex* start = incoming_begin(g, v);
        const Vertex* end = incoming_end(g, v);
        double sum = 0.0;
        for (const Vertex* vi = start ; vi != end ; vi++)
        {
          sum += score_old[*vi] / outgoing_size(g, *vi);
        }
        solution[v] = (damping * sum) + (1.0-damping) / numNodes + no_outgoing;
      }
      
      #pragma omp for reduction (+:global_diff)
      for (int v = 0 ; v < numNodes ; v++)
      {
        global_diff += fabs(solution[v] - score_old[v]);
      }
      
      converged = (global_diff < convergence);
      #pragma omp for 
      for (int vi = 0;vi < numNodes ; vi++)
      {
        score_old[vi] = solution[vi];
      }

    }
  }
  free(score_old);
  /*
     For PP students: Implement the page rank algorithm here.  You
     are expected to parallelize the algorithm using openMP.  Your
     solution may need to allocate (and free) temporary arrays.

     Basic page rank pseudocode is provided below to get you started:

     // initialization: see example code above
     score_old[vi] = 1/numNodes;

     while (!converged) {

       // compute score_new[vi] for all nodes vi:
       score_new[vi] = sum over all nodes vj reachable from incoming edges
                          { score_old[vj] / number of edges leaving vj  }
       score_new[vi] = (damping * score_new[vi]) + (1.0-damping) / numNodes;

       score_new[vi] += sum over all nodes v in graph with no outgoing edges
                          { damping * score_old[v] / numNodes }

       // compute how much per-node scores have changed
       // quit once algorithm has converged

       global_diff = sum over all nodes vi { abs(score_new[vi] - score_old[vi]) };
       converged = (global_diff < convergence)
     }

   */
}
