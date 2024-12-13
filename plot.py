import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation




# Define paths
convergence_path = "data/convergence/"
summary_path = "data/summary/results_summary.csv"
output_dir = "data/summary/plots/"
os.makedirs(output_dir, exist_ok=True)

# Plot convergence for each file, algorithm, and number of cities
def plot_convergence(convergence_path, output_dir):
    if not os.path.exists(convergence_path):
        print(f"Convergence directory not found: {convergence_path}")
        return
    
    convergence_files = [f for f in os.listdir(convergence_path) if f.endswith("_convergence.csv")]
    for file in convergence_files:
        data = pd.read_csv(os.path.join(convergence_path, file))
        plt.figure()
        plt.plot(data['Iteration'], data['Best Distance'], label=file.split('_')[1])  # Algorithm name
        plt.xlabel('Iteration')
        plt.ylabel('Best Distance')
        plt.title(f"Convergence: {file.split('_')[0]} ({file.split('_')[2]} cities)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, f"{file.split('.')[0]}_convergence.png"))
        plt.close()

# Plot execution times by number of cities
def plot_execution_times(summary_path, output_dir):
    if not os.path.exists(summary_path):
        print(f"Summary results file not found: {summary_path}")
        return

    summary_data = pd.read_csv(summary_path)
    plt.figure()
    for algorithm in summary_data['Algorithm'].unique():
        algo_data = summary_data[summary_data['Algorithm'] == algorithm]
        plt.plot(
            algo_data['Num Cities'], algo_data['Runtime'], label=algorithm, marker='o'
        )
    plt.xlabel('Number of Cities')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time by Number of Cities')
    plt.legend()
    plt.savefig(os.path.join(output_dir, "execution_times.png"))
    plt.close()


def create_comparison_gif(file, num_cities, coordinates_file, best_routes_file, output_path):
    # Load coordinates
    all_coordinates = np.loadtxt(coordinates_file, delimiter=' ', usecols=(1, 2))
    coordinates = all_coordinates[:num_cities]  # Use only the first `num_cities`

    # Load best routes
    best_routes = pd.read_csv(best_routes_file)
    subset = best_routes[(best_routes['File'] == file) & (best_routes['Num Cities'] == num_cities)]

    if subset.empty:
        print(f"No routes found for {file} with {num_cities} cities.")
        return

    # Extract routes for algorithms
    routes = {}
    for _, row in subset.iterrows():
        algorithm = row['Algorithm']
        route = list(map(int, row['Route'].split(',')))
        routes[algorithm] = route

    if len(routes) < 2:
        print(f"Not enough algorithms for comparison in {file} with {num_cities} cities.")
        return

    # Prepare for GIF
    algorithms = list(routes.keys())
    route1, route2 = routes[algorithms[0]], routes[algorithms[1]]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.plot(coordinates[:, 0], coordinates[:, 1], 'ko', markersize=5, label="Cities")

        # Plot Algorithm 1
        ordered_coords1 = coordinates[route1[:frame + 1]]
        ax.plot(ordered_coords1[:, 0], ordered_coords1[:, 1], 'o-', label=algorithms[0])

        # Plot Algorithm 2
        ordered_coords2 = coordinates[route2[:frame + 1]]
        ax.plot(ordered_coords2[:, 0], ordered_coords2[:, 1], 'x-', label=algorithms[1])

        ax.legend()
        ax.set_title(f"Comparison of Routes ({file}, {num_cities} Cities)\nStep {frame + 1}")

    total_duration = 10_000  # 10 seconds in milliseconds
    frames = max(len(route1), len(route2))  # Number of frames
    interval = total_duration / frames  # Milliseconds per frame

    ani = FuncAnimation(fig, update, frames=frames, interval=interval)
    ani.save(output_path, writer='imagemagick')
    plt.close()

    print(f"Comparison GIF saved at {output_path}")


if __name__ == "__main__":
    raw_path="data/raw/",
    best_routes_file="data/summary/best_routes.csv",
    output_dir="data/gifs/"    
    best_routes = pd.read_csv(best_routes_file)
    files = best_routes['File'].unique()

    for file in files:
        coordinates_file = os.path.join(raw_path, f"{file}.txt")
        if not os.path.exists(coordinates_file):
            print(f"Coordinates file missing for {file}: {coordinates_file}")
            continue

        # Process each subset of cities
        subsets = best_routes[best_routes['File'] == file]['Num Cities'].unique()
        for num_cities in subsets:
            output_path = os.path.join(output_dir, f"{file}_{num_cities}_comparison.gif")
            create_comparison_gif(file, num_cities, coordinates_file, best_routes_file, output_path)

    plot_convergence(convergence_path, output_dir)
    plot_execution_times(summary_path, output_dir)

    print(f"Plots saved in {output_dir}")
