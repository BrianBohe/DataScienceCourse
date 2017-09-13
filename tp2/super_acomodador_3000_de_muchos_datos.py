import scipy.io
import pandas as pd
import numpy as np
import gc


N_pacientes = 4
path = "/media/libre/los_datos_de_datos/calamaro.exp.dc.uba.ar/~fraimondo/cienciadatos/data/"
paciente_path = path + "P0{}.mat"
save_path = path + "dataframe_reconstruido_{}.pickle"
for paciente in range(1, N_pacientes + 1):
    print("Paciente {}".format(paciente))
    print("Levantando matriz...")
    mat = scipy.io.loadmat(paciente_path.format(paciente))['data']
    print("Listo")

    N_epochs = mat.shape[0]
    N_sensores = mat.shape[1]
    N_valores = mat.shape[2]

    N = N_epochs * N_sensores * N_valores

    print("Creando index...")
    iterables = [
        [paciente for _ in range(N)] ,
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
   
    print("Guardando pickle...")
    df_paciente.to_pickle(save_path.format(paciente))
    print("Listo")
    
    gc.collect()

