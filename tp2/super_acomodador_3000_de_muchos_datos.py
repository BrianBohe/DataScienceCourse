import scipy.io
import pandas as pd
import seaborn as sns
import numpy as np
import gc


N_pacientes = 1
path = "/home/francisco/tps/datos/tp2/~fraimondo/cienciadatos/data/P0{}.mat"
dfs = []
for paciente in range(1, N_pacientes + 1):
    print("Paciente {}".format(paciente))
    print("Levantando matriz...")
    mat = scipy.io.loadmat(path.format(paciente))['data']
    print("Listo")

    N_epochs = mat.shape[0]
    N_sensores = mat.shape[1]
    N_valores = mat.shape[2]

    N = N_epochs * N_sensores * N_valores

    iterables = [
        [paciente for _ in range(N)] ,
        [i for i in range(N_epochs) for j in range(N_sensores) for k in range(N_valores)],
        [j for i in range(N_epochs) for j in range(N_sensores) for k in range(N_valores)],
        [k for i in range(N_epochs) for j in range(N_sensores) for k in range(N_valores)],
    ]

    print("Creando index...")
    index = pd.MultiIndex.from_arrays(iterables, names=["paciente", "epoch", "sensor", "tiempo"])
    print("Listo")
    
    print("Haciendo reshape...")
    reshaped_mat = mat.reshape(N)
    print("Listo")

    print("Creando DataFrame...")
    dfs.append(pd.DataFrame({"valores": reshaped_mat}, index=index))
    print("Listo")

    gc.collect()
    
print("Concatenando dataframes...")
final_df = pd.concat(dfs)
print("Listo")
print("Guardando dataframe en formato hdf...")
final_df.to_hdf('dataframe_reconstruido.hdf', 'mis_datos', format='table', mode='w')
# Se levanta con pd.read_hdf('dataframe_reconstruido.hdf')
print("Listo")
