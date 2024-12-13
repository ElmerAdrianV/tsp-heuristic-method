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
import time
import csv
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

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

def two_opt_swap(route, i, j):
    return route[:i] + route[i:j+1][::-1] + route[j+1:]

def double_bridge_perturbation(route):
    n = len(route)
    p1, p2, p3, p4 = sorted(random.sample(range(1, n - 1), 4))
    return route[:p1] + route[p3:p4] + route[p2:p3] + route[p1:p2] + route[p4:]

def lin_kernighan_with_unstuck_refined(distance_matrix, start_city=0, max_iterations=1000, max_no_improvement=10):
    n = len(distance_matrix)
    route = [0] + random.sample(range(1, n), n - 1) + [0]
    best_route = route
    best_distance = calculate_total_distance(best_route, distance_matrix)
    print("Initial distance:", best_distance)

    no_improvement_count = 0
    perturbation_applied = False  # Track if perturbation has already been applied

    for iteration in range(max_iterations):
        improvement = False

        # 2-opt swaps
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_route = two_opt_swap(route, i, j)
                new_distance = calculate_total_distance(new_route, distance_matrix)

                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improvement = True
                    no_improvement_count = 0
                    perturbation_applied = False

        if improvement:
            route = best_route
        else:
            no_improvement_count += 1
            
            if not perturbation_applied:
                route = double_bridge_perturbation(best_route)
                perturbation_applied = True
            else:
                route = new_route 
            no_improvement_count = 0

        # Break if no improvement after perturbation and max iterations reached
        if perturbation_applied and no_improvement_count >= max_no_improvement:
            break

    return best_route, best_distance

def lin_kernighan(distance_matrix, start_city=0, max_iterations=1000, track_convergence=False):
    n = len(distance_matrix)
    route = [start_city] + random.sample(range(1, n), n - 1) + [start_city]
    best_route = route
    best_distance = calculate_total_distance(best_route, distance_matrix)
    print("Initial distance:", best_distance)

    convergence = []  # List to track the best distance at each iteration

    for iteration in range(max_iterations):
        improvement = False

        # 2-opt swaps
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_route = two_opt_swap(route, i, j)
                new_distance = calculate_total_distance(new_route, distance_matrix)

                if new_distance < best_distance:
                    best_route = new_route
                    best_distance = new_distance
                    improvement = True
                    print(f"Iteration {iteration}: Improved distance: {best_distance}")

        if track_convergence:
            convergence.append(best_distance)

        if improvement:
            route = best_route
        else:
            print(f"Iteration {iteration}: No improvement.")
            break  # Exit if no improvement is found

    if track_convergence:
        return best_route, best_distance, convergence
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

def load_coordinates(file_path):
    # Use numpy to load only the x and y coordinates (columns 1 and 2, zero-indexed)
    return np.loadtxt(file_path, delimiter=' ', usecols=(1, 2))


def save_convergence_data(convergence, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Iteration', 'Best Distance'])
        writer.writerows(enumerate(convergence))

def create_route_gif(route, coordinates, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ordered_coords = coordinates[route[:frame + 1]]
        ax.plot(ordered_coords[:, 0], ordered_coords[:, 1], 'o-')
        ax.set_title(f"Step {frame + 1}")
    
    # Total duration for the GIF
    total_duration = 10_000  # 10 seconds in milliseconds
    frames = len(route)  # Number of frames
    interval = total_duration / frames  # Milliseconds per frame

    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    ani.save(output_path, writer='imagemagick')

if __name__ == '__main__':
    # Experiment configuration
    files = ["Qatar", "Uruguay", "Zimbabwe"]
    raw_path = "data/raw/"
    matrix_path = "data/distance_matrix/"
    start_city = 0
    iterations = 1000
    subsets = [10, 50, 100, 194]  # Number of cities to test

    algorithms = {
        "Lin-Kernighan": lin_kernighan,
        "Christofides": christofides_algorithm
    }

    summary_results = []
    best_routes = []  # To store best route details

    for file in files:
        try:
            # Load city coordinates
            coordinates = load_coordinates(os.path.join(raw_path, f"{file}.txt"))
            total_cities = len(coordinates)
            print(f"Loaded coordinates for {file}: {total_cities} cities")

            # Ensure subsets do not exceed the total number of cities
            subsets_to_test = [s for s in subsets if s <= total_cities]

            # Load pre-generated distance matrix
            distance_matrix = np.genfromtxt(os.path.join(matrix_path, f"{file}_distances.csv"), delimiter=',')
            if np.isnan(distance_matrix).any():
                raise ValueError(f"Distance matrix in {file} contains invalid data.")

            for num_cities in subsets_to_test:
                subset_coords = coordinates[:num_cities]
                subset_matrix = distance_matrix[:num_cities, :num_cities]

                print(f"Processing {num_cities} cities for {file}")

                for name, algorithm in algorithms.items():
                    start_time = time.time()

                    if name == "Lin-Kernighan":
                        route, distance, convergence = algorithm(
                            subset_matrix, 
                            start_city=start_city, 
                            max_iterations=iterations,
                            track_convergence=True
                        )
                        save_convergence_data(
                            convergence, 
                            f"data/convergence/{file}_{name}_{num_cities}_convergence.csv"
                        )
                    else:
                        route, distance = algorithm(subset_matrix, start_city=start_city)

                    runtime = time.time() - start_time
                    summary_results.append([file, num_cities, name, distance, runtime])
                    best_routes.append({
                        'File': file,
                        'Num Cities': num_cities,
                        'Algorithm': name,
                        'Distance': distance,
                        'Route': route
                    })

                    print(f"{name} ({num_cities} cities) on {file}: Distance = {distance}, Runtime = {runtime:.2f}s")

                    create_route_gif(
                        route, 
                        subset_coords, 
                        f"data/gifs/{file}_{name}_{num_cities}_route.gif"
                    )

        except Exception as e:
            print(f"Error processing {file}: {e}")

    # Save summary results
    summary_csv_path = "data/summary/results_summary.csv"
    with open(summary_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['File', 'Num Cities', 'Algorithm', 'Final Distance', 'Runtime'])
        writer.writerows(summary_results)
    print(f"Summary results saved to {summary_csv_path}")

    # Save best routes
    best_routes_csv_path = "data/summary/best_routes.csv"
    with open(best_routes_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['File', 'Num Cities', 'Algorithm', 'Distance', 'Route'])
        for route_info in best_routes:
            writer.writerow([
                route_info['File'], 
                route_info['Num Cities'], 
                route_info['Algorithm'], 
                route_info['Distance'], 
                ','.join(map(str, route_info['Route']))
            ])
    print(f"Best routes saved to {best_routes_csv_path}")