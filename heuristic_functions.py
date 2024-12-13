"""
@ elmeradrianv

Este script implementa las funciones heuristicas para el TSP (Problema del Agente Viajero).
    - Chained Lin-Kernighan Heuristic
    - Christofides Algorithm
recibiendo como argumentos la matriz de distancias.
"""
import random
import numpy as np
import networkx as nx

def nearest_neighbor_solution(distance_matrix, start_city=0):
    n = len(distance_matrix)
    visited = [False] * n
    route = [start_city]
    visited[start_city] = True

    for _ in range(n - 1):
        last = route[-1]
        next_city = min(
            (i for i in range(n) if not visited[i]),
            key=lambda x: distance_matrix[last][x]
        )
        route.append(next_city)
        visited[next_city] = True

    route.append(start_city)  # Return to the start
    return route, calculate_total_distance(route, distance_matrix)

def calculate_total_distance(route, distance_matrix):
    return sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))

def lin_kernighan_improvement(route, distance_matrix):
    best_route = route
    best_distance = calculate_total_distance(route, distance_matrix)
    improved = True

    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_distance = calculate_total_distance(new_route, distance_matrix)
                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improved = True

    return best_route, best_distance

def perturb_tour(route):
    # Choose a random segment [i, j]
    i, j = sorted(random.sample(range(1, len(route) - 1), 2))

    segment = route[i:j+1]

    if random.choice([True, False]):
        segment = segment[::-1]
    
    new_route = route[:i] + route[j+1:] # Remove the segment from the route

    # Randomly choose a position to reinsert the segment
    insert_position = random.randint(1, len(new_route) - 1)
    perturbed_route = new_route[:insert_position] + segment + new_route[insert_position:]

    return perturbed_route

def chained_lin_kernighan(distance_matrix, start_city=0, iterations=10):
    current_route, _ = nearest_neighbor_solution(distance_matrix, start_city=start_city)
    improved_route, improved_distance = lin_kernighan_improvement(current_route, distance_matrix)
    
    # Initialize the best route and distance
    best_route = improved_route
    best_distance = improved_distance

    for _ in range(iterations):
        # Perturb the current best route
        perturbed_route = perturb_tour(improved_route)

        # Apply Lin-Kernighan improvements to the perturbed route
        improved_route, improved_distance = lin_kernighan_improvement(perturbed_route, distance_matrix)

        # Update the best route if the new improved route is better
        if improved_distance < best_distance:
            best_route = improved_route
            best_distance = improved_distance

    return best_route, best_distance


def christofides_algorithm(distance_matrix, start_city=0):
    n = len(distance_matrix)
    
    # Step 1: Construct the Minimum Spanning Tree (MST)
    G = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            G.add_edge(i, j, weight=distance_matrix[i][j])
    mst = nx.minimum_spanning_tree(G)

    # Step 2: Find vertices with odd degree in the MST
    odd_degree_vertices = [v for v in mst.nodes if mst.degree[v] % 2 == 1]

    # Step 3: Perform Minimum Weight Matching on odd-degree vertices
    subgraph = nx.Graph()
    for i in odd_degree_vertices:
        for j in odd_degree_vertices:
            if i != j:
                subgraph.add_edge(i, j, weight=distance_matrix[i][j])
    matching = nx.algorithms.matching.min_weight_matching(subgraph)

    # Step 4: Combine MST and Matching to form a multigraph
    multigraph = nx.MultiGraph()
    multigraph.add_edges_from(mst.edges)
    for u, v in matching:
        multigraph.add_edge(u, v, weight=distance_matrix[u][v])

    # Step 5: Find an Eulerian circuit
    eulerian_circuit = list(nx.eulerian_circuit(multigraph, source=start_city))

    # Step 6: Convert the Eulerian circuit to a Hamiltonian circuit
    visited = set()
    tour = []
    for u, v in eulerian_circuit:
        if u not in visited:
            tour.append(u)
            visited.add(u)
    tour.append(tour[0])  # Return to the starting city

    # Calculate total distance of the Hamiltonian circuit
    total_distance = sum(distance_matrix[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))

    return tour, total_distance

if __name__ == '__main__':
    # Ejemplo de uso
    distance_matrix = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0]
    ]

    final_route, final_distance = chained_lin_kernighan(distance_matrix, iterations=20)
    print("Ruta final con Chained Lin-Kernighan:", final_route)
    print("Distancia total con Chained Lin-Kernighan:", final_distance)

    
    tour, total_distance = christofides_algorithm(distance_matrix)

    print("Ruta final con Christofides Algorithm:", tour)
    print("Distancia total con Christofides Algorithm:", total_distance)