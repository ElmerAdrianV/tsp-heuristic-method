import heuristic_functions as hf

def experiment_1():
    """
    Experimento 1: Comparar la calidad de las soluciones obtenidas por los algoritmos
    de vecino más cercano y Lin-Kernighan en el problema del agente viajero.
    """
    files = ["Qatar", "Uruguay", "Zimbabwe"]
    path = "data/distance_matrix/"
    start_city = 0
    iterations = 10

    for file in files:
        distance_matrix = hf.read_csv(path + file + "_distances.csv")
        nn_route, nn_distance = hf.nearest_neighbor_solution(distance_matrix, start_city=start_city)
        nn_route, nn_distance = hf.lin_kernighan_improvement(nn_route, distance_matrix)
        lk_route, lk_distance = hf.chained_lin_kernighan(distance_matrix, start_city=start_city, iterations=iterations)

        print(f"Archivo: {file}")
        print(f"Vecino más cercano: {nn_distance}")
        print(f"Lin-Kernighan: {lk_distance}")
        print()

if