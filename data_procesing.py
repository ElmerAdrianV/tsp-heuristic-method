"""
@ elmeradrianv

Este script convierte un archivo de texto con las coordenadas de ciudades
en un archivo CSV que contiene una matriz de distancias euclidianas entre las ciudades, 
redondeadas al entero más cercano.
"""

import csv
import numpy as np
import os

def read_txt(file):
    """
    Lee un archivo de texto con ID y coordenadas de ciudades y devuelve una lista.
    """
    with open(file, 'r') as f:
        data = f.readlines()
    return data

def txt2csv(data, file):
    """
    Convierte una lista con las coordenadas de las ciudades
    en un archivo CSV
    """

    # Parse the data into a list of tuples
    data = [tuple(map(float, line_data.replace("\n", "").split(" ")[1:])) for line_data in data]
    print(data)
    coordinates = np.array(data)


    # Compute pairwise distances using broadcasting
    n = len(coordinates)
    x_coords = coordinates[:, 0].reshape(n, 1)
    y_coords = coordinates[:, 1].reshape(n, 1)

    distances = np.sqrt((x_coords - x_coords.T) ** 2 + (y_coords - y_coords.T) ** 2)
    distances = np.round(distances).astype(int) 

    current_dir = os.getcwd()  
    output_dir = os.path.join(current_dir, "data/distance_matrix")  # Target directory

    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

    # Save the distances to a CSV file in the specified directory
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(file))[0] + '_distances.csv')

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(distances)

if __name__ == '__main__':
    files_names = ["Qatar.txt", "Uruguay.txt", "Zimbabwe.txt"]
    path = "data/raw/"
    for file in files_names:
        full_path = os.path.join(path, file)
        print(f"Procesando archivo: {full_path}")
        data = read_txt(full_path)
        txt2csv(data, full_path)