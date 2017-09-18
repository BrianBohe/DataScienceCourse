import scipy.io
import pandas as pd
import numpy as np
import math
import gc
from collections import defaultdict


def transformacion_simbolica(df_):
    sigma = df_['valores'].std()
    min_v = min(df_['valores'])
    max_v = max(df_['valores'])
    
    # Calculo el N
    N = math.ceil((max_v - min_v) / (3.5 * sigma * len(df_['valores'])**(-1/3)))
    
    # Divido los valores en N rangos
    step = (max_v - min_v) / N
    ranges = []
    k = min_v
    for i in range(N):
        ranges.append((k, k + step))
        k += step
        
    ranges[-1] = (ranges[-1][0], ranges[-1][1] + step)
    apariciones = pd.Series([len(df_[(c_inf <= df_['valores']) & (df_['valores'] < c_sup)]) for c_inf, c_sup in ranges])
    
    df_t = pd.DataFrame({"cantidades": apariciones}) 
    return df_t

N_P = 1
N_S = 1

path = "/home/francisco/tps/datos/tp2/"
load_path = path + "{}.hdf"
save_path = path + "{}_transformado.hdf"
pacientes = \
        ["P{:02d}".format(i) for i in list(range(1, 1 + N_P))] + \
        ["S{:02d}".format(i) for i in list(range(1, 1 + N_S))]

# Primero encuentro el min y el max de TODAS las señales
datos = []
min_v = None
max_v = None
print("Buscando alfabeto...")
for paciente in pacientes:
    print("\tPaciente {}".format(paciente))
    print("\tLevantando hdf...")
    my_data = pd.read_hdf(load_path.format(paciente))
    print("\tListo")

    print("\tConsiguiendo data...")
    if max_v is None or max_v > max(my_data["valores"]):
        max_v = max(my_data["valores"])

    if min_v is None or min_v > min(my_data["valores"]):
        min_v = min(my_data["valores"])

    datos.append((len(my_data["valores"]), my_data["valores"].std(),))
    print("\tListo")
print("Listo")

print("Definiendo alfabeto...")
n, sigma = max(datos, key=lambda x: x[0])

N = math.ceil((max_v - min_v) / (3.5 * sigma * n ** (-1/3)))
step = (max_v - min_v) / N

def asociar_simbolo(v):
    return (v - min_v) // step
print("Listo")

total = 0
cantidades = defaultdict(int)
print("Calculando símbolo...")
for paciente in pacientes:
    print("\tPaciente {}".format(paciente))
    print("\tLevantando hdf...")
    my_data = pd.read_hdf(load_path.format(paciente))
    print("\tListo")

    print("\tAplicando transformación...")
    my_data["simbolo"] = asociar_simbolo(my_data["valores"])

    for s in my_data["simbolo"]:
        cantidades[s] += 1
        total += 1

    print("\tGuardando hdf...")
    my_data.to_hdf(save_path.format(paciente), "my_key", mode="w")
    print("\tListo")

    gc.collect()
print("Listo")

print("Calculando probabilidades para cada símbolo...")
for paciente in pacientes:
    print("\tPaciente {}".format(paciente))
    print("\tLevantando hdf...")
    my_data = pd.read_hdf(save_path.format(paciente))
    print("\tListo")
    
    print("\tCalculando probabilidad...")
    probs = []
    for s in my_data["simbolo"]:
        probs.append(cantidades[s] / total)
    my_data["probabilidad"] = np.array(probs)
    print("\tListo")

    print("\tGuardando hdf...")
    my_data.to_hdf(save_path.format(paciente), "my_key", mode="w")
    print("\tListo")
print("Listo")

