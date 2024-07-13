import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from time import time

start_time = time()

pic_size = (20, 20)
N_CONST = 1000
K_CONST = 20
num_of_graphs = 100
p_range = np.geomspace(start=0.0001, stop=1, num=num_of_graphs)

directory = f"plot_n={N_CONST}k={K_CONST}"

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


def create_small_world_graph(n, k, p):
    G = nx.watts_strogatz_graph(n, k, p)
    return G


data = []
for j in range(num_of_graphs):
    P_CONST = p_range[j]
    i = j + 1
    G = create_small_world_graph(N_CONST, K_CONST, P_CONST)

    avg_path_length = nx.average_shortest_path_length(G)
    diameter = nx.diameter(G)
    average_clustering = nx.average_clustering(G)
    data.append((avg_path_length, diameter, average_clustering))

path1 = os.path.join(directory, "L.png")
path2 = os.path.join(directory, "M.png")
path3 = os.path.join(directory, "C.png")


def plotter(path, title, x, y):
    plt.figure()
    plt.plot(x, y)
    plt.xscale("log")
    plt.title(title)
    plt.xlabel("p - logarithmic scale")
    plt.ylabel(title)
    plt.savefig(path)
    plt.close()


json_data = json.dumps(data)
plotter(path1, "Average path length", p_range, [x[0] for x in data])
plotter(path2, "Diameter", p_range, [x[1] for x in data])
plotter(path3, "Average clustering", p_range, [x[2] for x in data])

plot_data = {
    "P": p_range.tolist(),
    "L": [x[0] for x in data],
    "M": [x[1] for x in data],
    "C": [x[2] for x in data],
}
json_path = os.path.join(directory, "plot_data.json")
with open(json_path, "w") as json_file:
    json.dump(plot_data, json_file, indent=4)

end_time = time()
print(f"Execution time: {end_time - start_time:.2f} seconds")
print("_____________________________________\n")
