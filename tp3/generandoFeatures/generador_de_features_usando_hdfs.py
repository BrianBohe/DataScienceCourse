import pandas as pd
import numpy as np
import scipy.signal
import seaborn as sns
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import itertools
import math
import statistics as sts
import gc as gc

#Calculo la potencia de cada banda
def calcular_potencia_por_bandas_de_frecuencias(f, P):
    return {
        'delta': P[f<4].sum(),
        'theta': P[(4<=f) & (f<8)].sum(),
        'alpha': P[(8<=f) & (f<13)].sum(),
        'beta' : P[(13<=f) & (f<30)].sum(),
        'gamma': P[30<=f].sum()
    }

#  Dado un diccionario con pontencia de cada banda, 
# calculo la potencia normalizada de cada banda.
def normalizar_banda(bandas):
    suma_del_poder_total = sum(bandas.values()) 
    bandas_norm = {}
    for key, value in bandas.items():
        bandas_norm["{}_norm".format(key)]= bandas[key]/suma_del_poder_total
    return bandas_norm

# Calculo todos los marcadores que necesito para un paciente|epoch
# Estos son: bandas, bandas_norm, intra/inter(proximamente)
def calcular_marcadores_para_este_epoch(f,P):
    bandas = calcular_potencia_por_bandas_de_frecuencias(f,P)
    
    bandas_norm = normalizar_banda(bandas)
    bandas.update(bandas_norm)
    
    #Despues voy a ver que usamos de medida intra e interelectrodo
    bandas.update({"intra":np.random.rand(20,24), "inter":np.random.rand(20,24)})
    
    return bandas

def computar_features_para_estas_bandas(bandas_de_un_paciente):
    features_paciente = {}

    #Junto todos los valores de un mismo marcador para usar mean y stdev de statistics
    for key, value in bandas_de_un_paciente[0].items():
        
        _values = np.array([])
        
        for dic in bandas_de_un_paciente:
            _values = np.append(_values, dic[key])

        _mean = sts.mean(_values)
        _stdev = sts.stdev(_values,xbar=_mean)

        features_paciente["mean_{}".format(key)] = _mean
        features_paciente["stdev_{}".format(key)] = _stdev

    return features_paciente



def calcular_features_de_un_paciente(paciente):
    p_ = paciente.groupby(['epoch','tiempo']).mean()

    lista_de_epochs = list( set(p_.loc[:,:].index.get_level_values('epoch')) )
    bandas_de_este_paciente = []

    for epoch in lista_de_epochs:
        
        frecuencias = p_.loc[epoch,:]
        f, P = scipy.signal.welch(frecuencias['valores'], fs=250, nperseg=201)
        
        #Los marcadores segun el enunciado son bandas, bandas_normalizdas, 
        # algo inter, y algo intra electrodo
        bandas_de_este_paciente.append( calcular_marcadores_para_este_epoch(f,P) )

    #print (bandas_de_un_paciente)
    #print (bandas_de_un_paciente_norm)
    return computar_features_para_estas_bandas(bandas_de_este_paciente)


## Aprovechamos los .hdf generados a partir de los .mat en el tp anterior.
path = "../../tp2/datos/"
load_path = path + "/{}.hdf"

def levantar_hdf(load_name, nth):
    paciente = load_name + "{:02d}".format(i)
    return pd.read_hdf(load_path.format(paciente))


##Hay que preguntar si solo me interesan estos electrodos o todos
##Por ahora solo me quedo con los electrodos que me interesan
electrodos = [8, 44, 80, 131, 185]

N_P = 10
N_S = 10
features_P = []
features_S = []
for load_name, N, features, offset in [("P", N_P, features_P, 0), ("S", N_S, features_S, 10)]:
    for i in range(1, 1 + N):
        print(i)
        df_ = levantar_hdf(load_name, i)
        df_ = df_.loc[offset + i-1,:,electrodos,:]

        features.append( calcular_features_de_un_paciente(df_) )

        gc.collect()

print(features_P)
print("---")
print(features_S)

np_features = []

for features in [features_P, features_S]:
    for dic in features:
        np_aux = np.array(list(dic.values()))
        if np_features == []:
            np_features = np_aux
        else:
            np_features = np.vstack([np_features, [np_aux] ])

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

df = pd.DataFrame(data=np_features, index=index, columns=index_columns)

print(df)

df.to_pickle("../df_features.pickle")