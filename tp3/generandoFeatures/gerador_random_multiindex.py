import scipy.io
import pandas as pd
import numpy as np
import gc
import random

print("Creando index...")
arrays_index = [
    ['P']*10+['S']*10,
    [i for j in range(2) for i in range(10)]
]


index = pd.MultiIndex.from_arrays(arrays_index, names=["tipo", "indice_paciente"])
arrays_columns = [
    ['media']*12+['std']*12,
    ['delta', 'theta', 'alpha', 'beta', 'gamma', 'delta_norm', 'theta_norm', 'alpha_norm', 'beta_norm', 'gamma_norm', 'intra', 'inter']*2
]

index_columns = pd.MultiIndex.from_arrays(arrays_columns, names=["agrupacion_feature","feature"])

df = pd.DataFrame(data=np.array(np.random.rand(20,24)), index=index, columns=index_columns)

print(df)

df.to_pickle("/Users/bbohe/Documents/EXACTAS/DataScience/repo/tp3/df_features.pickle")
#sprint(index)