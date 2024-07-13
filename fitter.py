import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import math


# Load data from JSON files
def load_json_data(file_path, key):
    with open(file_path, "r") as file:
        data = json.load(file)
    return np.array(data["P"]), np.array(data[key])


# Example file paths
directory = "plot_avg_n=1000k=10"
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, directory)
file_path = os.path.join(folder_path, "plot_data.json")


# Define the model function
def exp_model(x, a, b, c):
    return a * (b**x) + c


def fitter(key):
    x_data, y_data = load_json_data(file_path, key)
    initial_guesses = [1.0, 0.1, 8.0]
    params, _ = curve_fit(exp_model, x_data, y_data, p0=initial_guesses, maxfev=20000)
    # a, b, c = 1, 2, 3
    a, b, c = params
    print(f"key: {key}, a: {a}, b: {b}, c: {c}")
    func = f"{a:.2f} * e^({math.log(b):.2f}x) + {c:.2f}"
    plt.scatter(x_data, y_data, label="Data")
    plt.plot(x_data, exp_model(x_data, a, b, c), color="red", label="Fit")
    plt.xlabel("p - logarithmic scale")
    if key == "L":
        plt.title(f"Average path length - {func}")
        plot_path = os.path.join(folder_path, "LD.png")
    elif key == "M":
        plt.title(f"Diameter - {func}")
        plot_path = os.path.join(folder_path, "MD.png")
    else:  # key == "C"
        plt.title(f"Average clustering - {func}")
        plot_path = os.path.join(folder_path, "CD.png")
    plt.legend()
    plt.savefig(plot_path)
    plt.close()


# fitter("C")
fitter("L")
# fitter("M")
