import scipy.io
import pandas as pd
import numpy as np
import gc


N_P = 2
N_S = 2

path = "/home/francisco/tps/datos/tp2/"
load_path = path + "{}.mat"
save_path = path + "{}.hdf"
pacientes = \
        ["P{:02d}".format(i) for i in list(range(1, 1 + N_P))] + \
        ["S{:02d}".format(i) for i in list(range(1, 1 + N_S))]

for paciente_index, paciente in enumerate(pacientes):
    print("Paciente {}".format(paciente))
    print("Levantando matriz...")
    mat = scipy.io.loadmat(load_path.format(paciente))['data']
    print("Listo")

    N_epochs = mat.shape[0]
    N_sensores = mat.shape[1]
    N_valores = mat.shape[2]

    N = N_epochs * N_sensores * N_valores

    print("Creando index...")
    iterables = [
        [paciente_index for _ in range(N)],
        [i for i in range(N_epochs) for j in range(N_sensores) for k in range(N_valores)],
        [j for i in range(N_epochs) for j in range(N_sensores) for k in range(N_valores)],
        [k for i in range(N_epochs) for j in range(N_sensores) for k in range(N_valores)],
    ]

    index = pd.MultiIndex.from_arrays(iterables, names=["paciente", "epoch", "sensor", "tiempo"])
    print("Listo")
    
    print("Haciendo reshape...")
    reshaped_mat = mat.reshape(N)
    print("Listo")

    print("Creando DataFrame...")
    df_paciente = pd.DataFrame({"valores": reshaped_mat}, index=index)
    print("Listo")
   
    print("Guardando hdf...")
    df_paciente.to_hdf(save_path.format(paciente), "my_key", mode="w")
    print("Listo")
    
    gc.collect()

