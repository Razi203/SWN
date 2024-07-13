import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from community import community_louvain
import numpy as np
import os
import glob
from time import time

start_time = time()

pic_size = (20, 20)
N_CONST = 800
K_CONST = 20
num_of_graphs = 20
p_range = np.geomspace(start=0.0001, stop=1, num=num_of_graphs)

directory = f"graphs_n={N_CONST}k={K_CONST}"

if not os.path.exists(directory):
    os.makedirs(directory)

print("\n_____________________________________")


# Delete all photos in a folder with specific extensions
def delete_all_photos(folder_path, extensions=[".png", ".jpg", ".jpeg", ".gif"]):
    for ext in extensions:
        files = glob.glob(os.path.join(folder_path, f"*{ext}"))
        for file in files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Error deleting {file}: {e}")


script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, directory)
delete_all_photos(folder_path)


# Create a Watts-Strogatz small-world graph
def create_small_world_graph(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    return G


def generate_graph(G, title, filename):
    if G is not None:
        partition = community_louvain.best_partition(G)
        node_size = [max(20 * nx.degree(G, node), 800) for node in G.nodes()]
        pos = nx.spring_layout(G)
        plt.figure(figsize=pic_size)  # Adjusted figure size for better visualization
        nx.draw_networkx_nodes(
            G,
            pos,
            partition.keys(),
            node_size=node_size,  # type: ignore
            cmap="rainbow",
            node_color=list(partition.values()),  # type: ignore
            alpha=0.8,
        )
        nx.draw_networkx_edges(G, pos, alpha=0.4)

        plt.title(title, fontsize=40)
        plt.savefig(filename)
        plt.close()
    else:
        print("Failed to create the graph: " + filename)


## Store graphs and their evaluation metrics
# graphs_data = []
counter = 0


# Wrapper function to store graphs and their metrics
def evaluate_and_store_graph(params):
    p = params[0]  # Rewiring probability
    k = K_CONST  # Fixed number of nearest neighbors
    n = N_CONST  # Fixed number of nodes
    G = create_small_world_graph(n, k, p)
    if G is None or not nx.is_connected(G):
        return 10**50  # Penalize disconnected or invalid graphs

    avg_path_length = nx.average_shortest_path_length(G)
    diameter = nx.diameter(G)
    average_clustering = nx.average_clustering(G)
    objective_value = +(avg_path_length + diameter) - average_clustering * 50
    ## Store the graph, parameters, and evaluation metrics
    # graphs_data.append((G, k, p, objective_value))
    global counter
    counter += 1
    return objective_value


# Define bounds for p (rewiring probability)
bounds = [(0.0001, 0.3)]

# Perform differential evolution optimization
result = differential_evolution(evaluate_and_store_graph, bounds)

optimize_time = time()

# Extract the best parameters
best_p = result.x[0]

print(f"n: {N_CONST}, k: {K_CONST} \nBest p: {best_p}")
print(f"Number of graphs evaluated: {counter}")

# Visualize the best small-world graph
best_G = create_small_world_graph(N_CONST, K_CONST, best_p)

print(f"connected: {nx.is_connected(best_G)}")
if nx.is_connected(best_G):
    avg_path_length = nx.average_shortest_path_length(best_G)
    diameter = nx.diameter(best_G)
    average_clustering = nx.average_clustering(best_G)
    print(f"average length: {avg_path_length:.2f}")
    print(f"diameter (max length): {diameter}")
    print(f"clustering: {average_clustering:.2f}")

    graph_data = (
        f"L = {avg_path_length:.2f} max = {diameter:.2f} C = {average_clustering:.2f}"
    )
    title = f"Optimized: n={N_CONST}, k={K_CONST}, p={best_p}\n " + graph_data
    path = os.path.join(directory, "optimized_swn.png")
    generate_graph(best_G, title, path)


for j in range(num_of_graphs):
    P_CONST = p_range[j]
    i = j + 1
    G = create_small_world_graph(N_CONST, K_CONST, P_CONST)

    graph_consts = f"n ={N_CONST} k = {K_CONST} p = {P_CONST}"
    avg_path_length = nx.average_shortest_path_length(G)
    diameter = nx.diameter(G)
    average_clustering = nx.average_clustering(G)
    graph_data = (
        f"L = {avg_path_length:.2f} max = {diameter:.2f} C = {average_clustering:.2f}"
    )
    path = os.path.join(directory, f"graph{i}.png")
    generate_graph(G, f"{graph_consts} \n {graph_data}", path)

end_time = time()

print(f"\nOptimization time: {optimize_time - start_time:.2f} seconds")
print(f"Execution time: {end_time - start_time:.2f} seconds")
print("_____________________________________\n")
