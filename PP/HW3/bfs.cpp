#include "bfs.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstddef>
#include <omp.h>
#include <limits.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list)
{
    list->count = 0;
}

void vertex_set_init(vertex_set *list, int count)
{
    list->max_vertices = count;
    list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
    vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(
    Graph g,
    vertex_set *frontier,
    vertex_set *new_frontier,
    int *distances)
{
    #pragma omp parallel
    {
        const int BUFFER_SIZE = 1024;
        int local_buffer[BUFFER_SIZE];
        int local_count = 0;

        // Parallelize the outer loop for processing frontier vertices
        #pragma omp for schedule(dynamic, 64)
        for (int i = 0; i < frontier->count; i++)
        {
            int node = frontier->vertices[i];
            int start_edge = g->outgoing_starts[node];
            int end_edge = (node == g->num_nodes - 1)
                              ? g->num_edges
                              : g->outgoing_starts[node + 1];

            // Process all neighbors of current node
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int outgoing = g->outgoing_edges[neighbor];
                
                // Use atomic compare-and-swap for thread-safe distance update
                if (__sync_bool_compare_and_swap(&distances[outgoing], 
                    NOT_VISITED_MARKER, 
                    distances[node] + 1))
                {
                    // Add to local buffer
                    local_buffer[local_count] = outgoing;
                    local_count++;

                    // If local buffer is full, flush to global frontier
                    if (local_count == BUFFER_SIZE)
                    {
                        #pragma omp critical
                        {
                            memcpy(&new_frontier->vertices[new_frontier->count],
                                  local_buffer,
                                  local_count * sizeof(int));
                            new_frontier->count += local_count;
                        }
                        local_count = 0;
                    }
                }
            }
        }

        // Final flush of remaining vertices in local buffer
        if (local_count > 0)
        {
            #pragma omp critical
            {
                memcpy(&new_frontier->vertices[new_frontier->count],
                      local_buffer,
                      local_count * sizeof(int));
                new_frontier->count += local_count;
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    // Initialize distances in parallel
    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // Setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {
#ifdef VERBOSE
        double start_time = CycleTimer::currentSeconds();
#endif

        vertex_set_clear(new_frontier);
        
        // Call parallel top_down_step
        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::currentSeconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }
}
void bottom_up_step(Graph graph, vertex_set *frontier, vertex_set *new_frontier, int *distances, int *height, bool *done, bool *front_inside, bool *new_front_inside)
{
    int local_count = 0;
    *done = true;

    #pragma omp parallel reduction(+:local_count)
    {
        // Each thread processes a subset of unvisited vertices
        #pragma omp for schedule(dynamic, 1024)
        for (int vid = 0; vid < graph->num_nodes; vid++) 
        {
            if (distances[vid] == -1)  // Node not yet visited
            {
                // Check incoming edges for any frontier node
                int edge_start = graph->incoming_starts[vid];
                int edge_end = (vid == graph->num_nodes - 1) ? 
                             graph->num_edges : 
                             graph->incoming_starts[vid + 1];

                for (int e = edge_start; e < edge_end; e++) 
                {
                    int parent = graph->incoming_edges[e];
                    if (front_inside[parent]) 
                    {
                        distances[vid] = *height;
                        new_front_inside[vid] = true;
                        local_count++;
                        *done = false;
                        break;  // Found a parent, no need to check other edges
                    }
                }
            }
        }
    }
    
    new_frontier->count = local_count;

}
void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    bool *front_inside = (bool*)malloc(sizeof(bool)*graph->num_nodes);
    bool *new_front_inside = (bool*)malloc(sizeof(bool)*graph->num_nodes);

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++) {
        sol->distances[i] = NOT_VISITED_MARKER; 
    }
    // Set up initial frontier with source node
    sol->distances[ROOT_NODE_ID] = 0;
    front_inside[ROOT_NODE_ID] = true;
    frontier->count = 1;
    
    int height = 1;
    bool done = false;
    
    // Main BFS loop
    while (!done) 
    {
        vertex_set_clear(new_frontier);
        
        // Process current frontier
        bottom_up_step(graph, 
                      frontier, 
                      new_frontier,
                      sol->distances,
                      &height,
                      &done,
                      front_inside,
                      new_front_inside);
        
        // Swap frontiers
        vertex_set *temp = frontier;
        frontier = new_frontier;
        new_frontier = temp;
        
        bool *tmp = front_inside;
        front_inside = new_front_inside;
        new_front_inside = tmp;

        height++;
    }
    
    // Cleanup
    free(frontier->vertices);
    free(new_frontier->vertices);
    free(front_inside);
    free(new_front_inside);
}
bool transfer (Graph graph, vertex_set *frontier)
{
    int mf = 0;
    for (int i=0;i<frontier->count;i++)
    {
        int node = frontier->vertices[i];
        mf += (node == graph->num_nodes-1) ? 0 : graph->outgoing_starts[node+1] - graph->outgoing_starts[node];
    }
    return (mf > 14 * graph->num_nodes);
}
void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    
    vertex_set list1;
    vertex_set list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    vertex_set *frontier = &list1;
    vertex_set *new_frontier = &list2;

    bool *front_inside = (bool*)malloc(sizeof(bool)*graph->num_nodes);
    bool *new_front_inside = (bool*)malloc(sizeof(bool)*graph->num_nodes);

    #pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;
    
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    front_inside[ROOT_NODE_ID] = true;
    sol->distances[ROOT_NODE_ID] = 0;
    
    bool done = false;
    int height = 1;
    bool top_down_turn = true;
    while(!done && frontier->count!=0)
    {
        vertex_set_clear(new_frontier);
        if (top_down_turn && transfer(graph, frontier))
            top_down_turn = false;
        else if (!top_down_turn && frontier->count < graph->num_nodes / 24)
            top_down_turn = true;
        if (top_down_turn)
        {
            // do top down
            top_down_step(graph, frontier, new_frontier, sol->distances);
            done = (new_frontier->count == 0);
        }else
        {
            bottom_up_step(graph, frontier, new_frontier, sol->distances, &height, &done, front_inside, new_front_inside);
            bool *tmp = front_inside;
            front_inside = new_front_inside;
            new_front_inside = tmp;
        }
        vertex_set *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
        
        height++;
    }
    free(frontier->vertices);
    free(new_frontier->vertices);
    free(front_inside);
    free(new_front_inside);
}




