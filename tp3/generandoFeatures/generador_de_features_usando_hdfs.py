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
from sklearn.preprocessing import StandardScaler

##Definimos el orden de los atributos antes de codear de forma rapida
##Esta clase esta solo para asegurar ese orden.
class Sample():

    def __init__(self, featuresDict):
        self.features = {}

        for k,v in featuresDict.items():
            self.features[k] = v

    def show_features_as_np(self):
        valores = np.empty([1,24])
        
        for i, forma_de_calcular_feature in enumerate(['media_','std_']):
            for  j, tipo_de_banda in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma', 'delta_norm', 'theta_norm', 'alpha_norm', 'beta_norm', 'gamma_norm', 'intra', 'inter']):
                valores[i*12+j] = self.features["{}{}".format(forma_de_calcular_feature,tipo_de_banda)]

        return valores

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


def crear_sample_con_features(lista_de_marcadores_por_epoch):
    features_paciente = {}

    #Junto todos los valores de un mismo marcador para usar mean y stdev de statistics
    for key, value in lista_de_marcadores_por_epoch[0].items():
        
        _values = np.array([])
        
        for dic in lista_de_marcadores_por_epoch:
            _values = np.append(_values, dic[key])

        _mean = sts.mean(_values)
        _stdev = sts.stdev(_values,xbar=_mean)

        features_paciente["media_{}".format(key)] = _mean
        features_paciente["std_{}".format(key)] = _stdev

    return Sample(features_paciente)



def calcular_features_de_un_paciente(paciente):
    p_ = paciente.groupby(['epoch','tiempo']).mean()

    lista_de_epochs = list( set(p_.loc[:,:].index.get_level_values('epoch')) )
    lista_de_marcadores_por_epoch = []

    for epoch in lista_de_epochs:
        
        frecuencias = p_.loc[epoch,:]
        f, P = scipy.signal.welch(frecuencias['valores'], fs=250, nperseg=201)
        

        # Calculo todos los marcadores que necesito para un epoch
        # Estos son las bandas, las bandas normalizadas y alguna medida intra/inter electrodo
        marcadores_de_este_epoch = calcular_potencia_por_bandas_de_frecuencias(f,P)
        marcadores_de_este_epoch.update(normalizar_banda(marcadores_de_este_epoch))
        
        #TODO: Cambio de Intra Inter
        marcadores_de_este_epoch.update({"intra":np.random.rand(20,24), "inter":np.random.rand(20,24)})
        
        lista_de_marcadores_por_epoch.append( marcadores_de_este_epoch )

    return crear_sample_con_features(lista_de_marcadores_por_epoch)


## Aprovechamos los .hdf generados a partir de los .mat en el tp anterior.
path = "../../tp2/datos/"
load_path = path + "/{}.hdf"

def levantar_hdf(load_name, nth):
    paciente = load_name + "{:02d}".format(i)
    return pd.read_hdf(load_path.format(paciente))


##Hay que preguntar si solo me interesan estos electrodos o todos
##Por ahora solo me quedo con los electrodos que me interesan
electrodos = [8, 44, 80, 131, 185]

N_P = 1
N_S = 1

lista_de_muestras = []

for load_name, N, offset in [("P", N_P, 0), ("S", N_S, 10)]:
    for i in range(1, 1 + N):
        print(i)
        df_ = levantar_hdf(load_name, i)
        df_ = df_.loc[offset + i-1,:,electrodos,:]

        lista_de_muestras.append( calcular_features_de_un_paciente(df_) )

        gc.collect()


#Standarizo los datos
print(lista_de_muestras)

np_features_por_paciente = []
for muestra in lista_de_muestras:
    np_features_por_paciente.append( muestra.show_features_as_np() )

print(np_features_por_paciente)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
scaler.fit(np_features_por_paciente)
np_features_por_paciente = scaler.transform(np_features_por_paciente)

print(np_features_por_paciente)

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

df = pd.DataFrame(data=np_features_por_paciente, index=index, columns=index_columns)

print(df)

df.to_pickle("../df_features.pickle")