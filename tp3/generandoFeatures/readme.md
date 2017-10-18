

```python
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
import sys
from sklearn import preprocessing


## Aprovechamos los .hdf generados a partir de los .mat en el tp anterior.
path = "../../tp2/datos/"
load_path = path + "/{}.hdf"

##Electrodos de interes
electrodos = [8, 44, 80, 131, 185]

# Cantidad de pacientes S y P
N_P = 10
N_S = 10
```

<h1>Script Generador de Features</h1>
<blockquote>
    <p>Este script genera los features que luego utilizaremos en el análisis univariado y multivariado.</p>
    <p>Estas son:
    <ul>
        <li>Potencia para cada banda de frecuencia (Delta, Theta, Alpha, Beta y Gamma)</li>
        <li>Potencia normalizada para las mismas bandas de frecuencia.</li>
        <li>Una medida de información intra-electrodo</li>
        <li>Una medida de información inter-electrodo</li>
    </ul>
    </p>
    <p>Cada marcador se computa por cada epochs y luego se toma la media y el desvío estándard entre los valores de cada epochs. En el caso de los features espectrales, para el cómputo de un epoch primero se promedian las frecuencias entre los mismos electrodos utilizados en el TP2.</p>
    <p>Como medida intra-electrodo se decide utilizar la fuente de información modelada en el TP2 considerando sólo al electrodo 8. Se decide esto porque en los resultados observados presentó una mejor diferencia en los rangos de entropía para entre los pacientes P y S.</p>
    <p>La medida inter-electrodo es la entropía de la fuente modelada con un alfabeto general para todos los electrodos (lo mismo que se utilizó en el [TP2](https://github.com/BrianBohe/DataScienceCourse/tree/master/tp2).</p>
</blockquote>

<h4>Construyendo alfabetos</h4>
<blockquote>
<p>Primero vamos a generar al igual que en el TP anterior los alfabetos que se utilzarán luego en las medidas inter/intra-electrodo.</p>
</blockquote>


```python
pacientes_P = []
pacientes_S = []
for load_name, N, dest, offset in [("P", N_P, pacientes_P, 0), ("S", N_S, pacientes_S, 10)]:
    for i in range(1, 1 + N):
        paciente = load_name + "{:02d}".format(i)
        df_ = pd.read_hdf(load_path.format(paciente))
        df_ = df_.loc[offset + i-1,:,electrodos,:]
        dest.append(df_)

pd_pacientes = pd.concat(pacientes_P + pacientes_S)
```


```python
pacientes_b_mean = []
N_b = 0
step_b = 0
alfabeto_8 = []

# Calculo de N (cantidad de bins): Se usa n = 201 porque los epochs tienen esa cantidad de muestras
def calculo_N(df_):
    return math.ceil((df_.max() - df_.min()) / (3.5*df_.std() * 201 ** (-1/3)))

def calcular_simbolos_sensor_8_para_promediando_entre_pacientes(pacientes):
    pacientes_8 = pacientes.loc[pd.IndexSlice[:,:,8,:],:]
    val = pacientes_8.groupby(["epoch"])['valores'].mean()
    i_s = pd.DataFrame({'sensor':[8]})
    i_s['N'] = calculo_N(val)
    i_s['min'] = val.min()
    i_s['max'] = val.max()
    i_s['step'] = (i_s['max'] - i_s['min']) / i_s['N']
    return i_s

# Calcular N para hacer la transformación simbólica
def calcular_simbolos_inter_promediando_entre_pacientes(pacientes):
    pacientes_b_mean = pd.concat(pacientes_P + pacientes_S).groupby(["sensor","tiempo"]).mean()
    N_b = math.ceil(pacientes_b_mean.max() - pacientes_b_mean.min() / (pacientes_b_mean.std() * len(pacientes_b_mean) ** (-1/3)))
    step_b = (pacientes_b_mean.max() - pacientes_b_mean.min()) / N_b
    return pacientes_b_mean, N_b, step_b

def calcular_entropia_inter_electrodo_para_un_epoch(df_):
    df_['simbolos_inter'] = (df_["valores"] - pacientes_b_mean['valores'].min()) // step_b['valores']
    df_[df_['simbolos_inter'] < 0] = 0
    df_[df_['simbolos_inter'] > N_b] = N_b
    df_['repeticiones_inter'] = df_.groupby(["simbolos_inter"]).transform('count')['valores']
    df_['probabilidad_inter'] = df_['repeticiones_inter'] / len(df_)
    p = df_["probabilidad_inter"]
    return -sum(p * np.log(p))
    
def calcular_entropia_para_un_epoch(df_):
    df_['simbolos_intra'] = (df_['valores'].values - alfabeto_8['min'].values) // alfabeto_8['step'].values
    df_[df_['simbolos_intra'] < 0] = 0
    df_[df_['simbolos_intra'] > alfabeto_8['N'].values[0]] = alfabeto_8['N'].values[0]
    df_['repeticiones_intra'] = df_.groupby(["simbolos_intra"]).transform('count')['valores']
    df_['probabilidad_intra'] = df_['repeticiones_intra'] / len(df_)
    df_ = df_.groupby("simbolos_intra").first()
    p = df_["probabilidad_intra"]
    return -sum(p * np.log(p))

# Se calcula la cantidad de Bins promediando entre todos los pacientes
alfabeto_8 = calcular_simbolos_sensor_8_para_promediando_entre_pacientes(pd_pacientes)

# Para el caso de la medida inter-electrodo se calcula la cantidad de Bins entre todos los electrodos y pacientes 
pacientes_b_mean, N_b, step_b = calcular_simbolos_inter_promediando_entre_pacientes(pd_pacientes)
```


```python
## Establecimos este orden para los features
def give_expected_ordered_keys():
    ordered_keys = []
    for i, forma_de_calcular_feature in enumerate(['media_','std_']):
        for  j, tipo_de_banda in enumerate(['delta', 'theta', 'alpha', 'beta', 'gamma', 'delta_norm', 'theta_norm', 'alpha_norm', 'beta_norm', 'gamma_norm', 'intra', 'inter']):
            ordered_keys.append("{}{}".format(forma_de_calcular_feature,tipo_de_banda))
    return np.array(ordered_keys)

##  Y así será al forma de los archivos pickle generados con los features (en crudo y normalizadas)
def numpy_to_pickle(aCollectionOfNumpyArray, aPickelFileName):
    print("Creando index...")
    lenght = int(len(aCollectionOfNumpyArray)/2)
    arrays_index = [
        ['P']*N_P+['S']*N_S,
        [i for j in range(2) for i in range(lenght)]
    ]

    index = pd.MultiIndex.from_arrays(arrays_index, names=["tipo", "indice_paciente"])
    arrays_columns = [
        ['media']*12+['std']*12,
        ['delta', 'theta', 'alpha', 'beta', 'gamma', 'delta_norm', 'theta_norm', 'alpha_norm', 'beta_norm', 'gamma_norm', 'intra', 'inter']*2
    ]

    index_columns = pd.MultiIndex.from_arrays(arrays_columns, names=["agrupacion_feature","feature"])

    df = pd.DataFrame(data=aCollectionOfNumpyArray, index=index, columns=index_columns)

    print(df)

    df.to_pickle("../data_set/{}.pickle".format(aPickelFileName))
```


```python
##Definimos el orden de los atributos antes de codear de forma rapida
##Esta clase esta solo para asegurar ese orden.
class Sample():

    def __init__(self, featuresDict):
        self.features = {}

        for k,v in featuresDict.items():
            self.features[k] = v
                
    def show_features_as_np(self):
        valores = []
        for key in give_expected_ordered_keys():
            valores.append(self.features["{}".format(key)])
        
        return np.array(valores)
```


```python
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
#      calculo la potencia normalizada de cada banda.
def normalizar_banda(bandas):
    suma_del_poder_total = sum(bandas.values()) 
    bandas_norm = {}

    for key, value in bandas.items():
        bandas_norm["{}_norm".format(key)]= bandas[key]/suma_del_poder_total

    return bandas_norm

def computar_promedio_y_desvio_por_marcador_sobre_todos_los_epochs(lista_de_marcadores_por_epoch):
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

# De cada muestra o sample tomamos 24 features ya elegidos, computados a través del promedio y varianza
#      entre los epochs de cada paciente.
def calcular_features_de_un_paciente(paciente):
    
    marcadores_de_este_epoch = {}

    ## Junto la informacion por cada epoch
    p_ = paciente.groupby(['epoch','tiempo']).mean()
    
    lista_de_epochs = list( set(p_.loc[:,:].index.get_level_values('epoch',)) )
    lista_de_marcadores_por_epoch = []
    
    for epoch in lista_de_epochs:
        
        frecuencias = p_.loc[epoch,:]
        f, P = scipy.signal.welch(frecuencias['valores'], fs=250, nperseg=201)
        
        # Calculo todos los marcadores que necesito para un epoch
        # Estos son las bandas, las bandas normalizadas y alguna medida intra/inter electrodo
        marcadores_de_este_epoch = calcular_potencia_por_bandas_de_frecuencias(f,P)
        marcadores_de_este_epoch.update(normalizar_banda(marcadores_de_este_epoch))
        
        # Para calcular el marcador intra-electrodo solo necesito la frecuencia del sensor/electrodo 8
        frecuencias_8 = paciente.loc[pd.IndexSlice[:,epoch,8,:],:].groupby(['epoch','tiempo']).mean()
        marcadores_de_este_epoch.update({"intra":calcular_entropia_para_un_epoch(frecuencias_8)})
        
        #TODO: Cambio de Intra Inter
        marcadores_de_este_epoch.update({"inter":calcular_entropia_inter_electrodo_para_un_epoch(frecuencias)})
        lista_de_marcadores_por_epoch.append( marcadores_de_este_epoch )
    
    ## Ahora calculo los features tomando promedio y desvio sobre los marcadores de cada epoch
    sample = computar_promedio_y_desvio_por_marcador_sobre_todos_los_epochs(lista_de_marcadores_por_epoch)

    return sample

#   Usamos los mismos archivos *.hdf generados para el TP anterior.
def levantar_hdf(load_name, nth):
    paciente = load_name + "{:02d}".format(i)
    return pd.read_hdf(load_path.format(paciente))
```


```python
samples_pacientes = []

pacientes = list(pacientes_P + pacientes_S)
for paciente in pacientes:
    samples_pacientes.append(calcular_features_de_un_paciente(paciente))
```


```python
samples_pacientes[0].show_features_as_np()
```




    array([  1.64020028e-11,   3.21256487e-12,   1.06260520e-12,
             1.09448768e-12,   2.44176599e-13,   6.38981501e-01,
             1.78470984e-01,   8.12785435e-02,   7.63432110e-02,
             2.49257608e-02,   2.14470985e+00,   5.08732756e+01,
             2.55164885e-11,   4.31202688e-12,   1.13895601e-12,
             1.45936210e-12,   2.96494982e-13,   1.92985867e-01,
             1.29179819e-01,   8.19777064e-02,   7.18567245e-02,
             3.21789162e-02,   2.54985457e-01,   1.03480686e+01])




```python
len(list(pacientes_P + pacientes_S))
```




    20



Miremos como están quedando los datos. Un buen grafico para ver que tan esparsos son los valores de cada feature estaría bueno.


```python
#Codigo para imprimir como barplot, violinplot y swarmplot un DataFrame
def analisis_comparativo(df_banda, df_banda_estandarizado):
#    ymin = min(df_banda['Valores'])
#    ymax = max(df_banda['Valores'])
#    decimo = (ymax - ymin)/len(df_banda['Valores'])
#    ymin, ymax = ymin - decimo, ymax + decimo

    # Hay que tener en cuenta que son pocos valores

    #ax = sns.violinplot(x="Features", y="Valores", hue="Capacidad cognitiva", data=df_b,  split=True, palette="Set2", inner="stick", cut=0)
    #sns.plt.show()
    fig, (ax1,ax2) = plt.subplots(1,2, sharey=True, figsize=(12,4))
    
    #ax1
#    sns.pointplot("who", "age", data=titanic, join=False,n_boot=10, ax=ax1)

    sns.swarmplot(x="Sin Estandarizar", y="Valores", hue="Capacidad cognitiva",data=df_banda,  split=True, palette="Set2", size=4, ax=ax1)
#    ax1.set_ylim([ymin, ymax])
    
    sns.swarmplot(x="Estandarizado", y="Valores", hue="Capacidad cognitiva", data=df_banda_estandarizado,  split=True, palette="Set2", size=4, ax=ax2)
    sns.plt.show()

    #sns.barplot(x="Features", y="Valores", hue="Capacidad cognitiva", data=df_b, palette="Set2")
    #ax = sns.swarmplot(x="Features", y="Valores", hue="Capacidad cognitiva", split=True, data=df_b, palette="Set2")
    #ax.set_ylim([ymin, ymax])
    
    #handles, labels = ax.get_legend_handles_labels()
    #l = ax.legend(handles[:2], labels[:2])
    #l.set_title("Capacidad cognitiva", prop = {'size':'small'})
    #sns.plt.show() 
```


```python
# imprimo por feature para ver como quedaron los datos
def printear_comparacion_sin_y_con_estandarizar(nd_datos, nd_datos_stand, lista_de_keys):
    
    for indice, key in enumerate(lista_de_keys):
        valores = list([nd[indice] for nd in nd_datos])
        
        valores_estandarizados = list([nd[indice] for nd in nd_datos_stand])
        
        keys = [key]* (N_P+N_S)

        df_banda = pd.DataFrame({
            "Capacidad cognitiva": (["Disminuída"] * N_P) + (["Normal"] * N_S),
            "Sin Estandarizar": keys,
            "Valores": valores
        })
        
        df_banda_estandarizada = pd.DataFrame({
            "Capacidad cognitiva": (["Disminuída"] * N_P) + (["Normal"] * N_S),
            "Estandarizado": keys,
            "Valores": valores_estandarizados
        })
        #print ("resultado \n")
        #print (df_for_one_feature)
        analisis_comparativo(df_banda, df_banda_estandarizada)
```


```python
np_features_por_paciente = []
for muestra in samples_pacientes:
    np_features_por_paciente.append( muestra.show_features_as_np() )
```


```python
# todas las keys encadenadas
#print (len ( list(itertools.chain(*[list(give_expected_ordered_keys()) for _ in range(N_P+N_S)]))) )
# todos los valores encadenados
#print (len(list(itertools.chain(*[list(paciente) for paciente in np_features_por_paciente]))) )
# todos los headers encadenados
#print (len( (["Reducida"] * (N_P * 24)) + (["Normal"] * (N_S * 24))) )

#df_features_compress = pd.DataFrame({
#    "Capacidad cognitiva": (["Reducida"] * (N_P * 24)) + (["Normal"] * (N_S * 24)),
#    "Features": list(itertools.chain(*[list(give_expected_ordered_keys()) for _ in range(N_P+N_S)])),
#    "Valores": list(itertools.chain(*[list(paciente) for paciente in np_features_por_paciente]))
#})
```


```python
np_features_por_paciente
```




    [array([  1.64020028e-11,   3.21256487e-12,   1.06260520e-12,
              1.09448768e-12,   2.44176599e-13,   6.38981501e-01,
              1.78470984e-01,   8.12785435e-02,   7.63432110e-02,
              2.49257608e-02,   2.14470985e+00,   5.08732756e+01,
              2.55164885e-11,   4.31202688e-12,   1.13895601e-12,
              1.45936210e-12,   2.96494982e-13,   1.92985867e-01,
              1.29179819e-01,   8.19777064e-02,   7.18567245e-02,
              3.21789162e-02,   2.54985457e-01,   1.03480686e+01]),
     array([  2.02727252e-11,   4.90009184e-12,   7.60557088e-13,
              6.96469623e-13,   3.01721905e-13,   6.88463437e-01,
              2.16743736e-01,   3.84609745e-02,   3.90183085e-02,
              1.73135433e-02,   2.30183262e+00,   5.16707332e+01,
              2.44120853e-11,   4.66768229e-12,   8.31035167e-13,
              5.95301344e-13,   2.46609858e-13,   1.97354017e-01,
              1.65658943e-01,   3.74246957e-02,   3.89387071e-02,
              1.87171982e-02,   2.02654002e-01,   1.11734374e+01]),
     array([  2.93307575e-11,   5.28851723e-12,   1.23387513e-12,
              6.21889752e-13,   1.24701150e-13,   7.47843982e-01,
              1.70893982e-01,   4.99085189e-02,   2.59690648e-02,
              5.38445269e-03,   2.34431909e+00,   5.11919258e+01,
              2.44839923e-11,   4.95270068e-12,   9.17576450e-13,
              2.96333776e-13,   5.87534668e-14,   1.52883084e-01,
              1.20008654e-01,   4.86548203e-02,   2.17308188e-02,
              4.99831596e-03,   2.39313986e-01,   1.23788964e+01]),
     array([  1.83300729e-11,   6.41798254e-12,   2.20051565e-12,
              9.27360649e-13,   9.50521643e-14,   5.85925873e-01,
              2.66792376e-01,   9.69220638e-02,   4.53772475e-02,
              4.98244029e-03,   2.29954756e+00,   5.20287633e+01,
              1.84127325e-11,   5.56743117e-12,   2.13741245e-12,
              7.93770354e-13,   1.02198544e-13,   2.17678082e-01,
              1.75357794e-01,   8.84973674e-02,   4.27225554e-02,
              5.77722731e-03,   2.11213605e-01,   1.11846605e+01]),
     array([  2.49446658e-11,   5.18365108e-12,   1.53549593e-12,
              1.09063309e-12,   4.40644066e-13,   6.84974822e-01,
              1.89109380e-01,   6.11715373e-02,   4.55922414e-02,
              1.91520189e-02,   2.34922692e+00,   5.27438076e+01,
              1.93503606e-11,   4.01637213e-12,   1.06963095e-12,
              5.48188989e-13,   2.80422397e-13,   1.85923239e-01,
              1.41364621e-01,   5.30083364e-02,   3.80786229e-02,
              1.94640841e-02,   1.92877827e-01,   1.14677076e+01]),
     array([  1.97445003e-11,   3.56000123e-12,   2.80610332e-12,
              1.01153553e-12,   7.72459156e-14,   6.50266341e-01,
              1.61244803e-01,   1.29734946e-01,   5.45366482e-02,
              4.21726128e-03,   2.31127395e+00,   5.11254387e+01,
              1.94039226e-11,   3.36045955e-12,   2.56494612e-12,
              8.71674200e-13,   4.99810663e-14,   1.98293796e-01,
              1.28831061e-01,   1.09324175e-01,   5.63218078e-02,
              3.89874074e-03,   2.17434890e-01,   1.16045951e+01]),
     array([  1.84595979e-11,   3.06384385e-12,   6.71048215e-13,
              8.35989566e-13,   3.76620976e-13,   7.20758908e-01,
              1.57006768e-01,   4.32822184e-02,   5.55330543e-02,
              2.34190512e-02,   2.29552947e+00,   5.26216669e+01,
              1.62819324e-11,   2.53585390e-12,   4.87710097e-13,
              6.37172203e-13,   6.01441334e-13,   1.69481010e-01,
              1.09361551e-01,   4.30950877e-02,   5.58167959e-02,
              3.52016097e-02,   2.04486154e-01,   1.01640574e+01]),
     array([  7.32008586e-12,   2.17422069e-12,   4.14155562e-13,
              3.57699873e-13,   1.56939936e-13,   6.17072216e-01,
              2.52727480e-01,   5.56992427e-02,   5.22160005e-02,
              2.22850610e-02,   2.05550185e+00,   5.06344112e+01,
              9.54042370e-12,   2.09660738e-12,   4.43925826e-13,
              2.83836157e-13,   2.04503995e-13,   2.16568199e-01,
              1.77382200e-01,   5.30245998e-02,   4.32761122e-02,
              2.58702236e-02,   2.63663974e-01,   8.97129842e+00]),
     array([  1.34070686e-11,   5.33836179e-12,   9.89457744e-13,
              9.78213580e-13,   1.83071914e-13,   5.90541926e-01,
              2.71263919e-01,   6.28311389e-02,   6.23128913e-02,
              1.30501252e-02,   2.24375965e+00,   5.34617005e+01,
              1.19495980e-11,   4.91443688e-12,   8.62196600e-13,
              8.73599208e-13,   1.16821704e-13,   1.99266272e-01,
              1.68754151e-01,   5.53579198e-02,   6.20978594e-02,
              1.22231751e-02,   2.14571428e-01,   9.76858902e+00]),
     array([  2.19527438e-11,   3.88026087e-12,   1.06626220e-12,
              1.11237073e-12,   2.14422246e-13,   7.06442744e-01,
              1.71451877e-01,   5.45985694e-02,   5.59537779e-02,
              1.15530321e-02,   2.34263296e+00,   5.04497254e+01,
              2.10961068e-11,   4.30661746e-12,   8.78047406e-13,
              1.02857049e-12,   2.19653631e-13,   1.82768342e-01,
              1.39415087e-01,   5.18226774e-02,   5.26289704e-02,
              1.33853679e-02,   2.11949054e-01,   1.19720623e+01]),
     array([  8.78583783e-12,   2.25518683e-12,   2.06168893e-12,
              5.39390326e-12,   6.34646549e-12,   3.21599087e-01,
              9.54962143e-02,   8.97802708e-02,   2.26481929e-01,
              2.66642499e-01,   2.38094767e+00,   5.62823415e+01,
              9.15337668e-12,   2.17836585e-12,   1.89874112e-12,
              4.21250004e-12,   4.43018868e-12,   1.74887228e-01,
              7.13911894e-02,   6.86617481e-02,   1.11956304e-01,
              1.20389222e-01,   1.78849575e-01,   7.92658374e+00]),
     array([  3.27255133e-12,   2.44078916e-12,   6.83530254e-12,
              1.60064461e-12,   1.13043451e-13,   2.26933344e-01,
              1.72814443e-01,   4.58728157e-01,   1.31342512e-01,
              1.01815433e-02,   2.20819678e+00,   5.76015786e+01,
              4.83484191e-12,   2.48548631e-12,   5.07266478e-12,
              9.26086894e-13,   8.42378166e-14,   1.61203189e-01,
              1.29341985e-01,   1.96878760e-01,   8.39562573e-02,
              1.20501569e-02,   1.93725631e-01,   6.39600966e+00]),
     array([  3.92576050e-12,   2.72270757e-12,   1.24135860e-11,
              2.18203641e-12,   1.74600976e-13,   2.10915112e-01,
              1.39672419e-01,   5.15821829e-01,   1.22847265e-01,
              1.07433747e-02,   2.36004574e+00,   5.86603717e+01,
              3.54454483e-12,   2.89886791e-12,   1.13155220e-11,
              1.30777652e-12,   1.01007313e-13,   1.53438863e-01,
              1.00724658e-01,   2.05763061e-01,   7.36356515e-02,
              8.09318878e-03,   2.19512490e-01,   6.59492928e+00]),
     array([  5.81786189e-12,   2.16757512e-12,   7.26102487e-12,
              2.88961325e-12,   1.94837679e-12,   2.65521727e-01,
              1.16801316e-01,   3.82267050e-01,   1.45387946e-01,
              9.00219610e-02,   2.22962326e+00,   5.69184689e+01,
              9.33684664e-12,   2.01063380e-12,   5.30421524e-12,
              3.32899540e-12,   2.92392446e-12,   1.66511025e-01,
              7.95603682e-02,   1.76614230e-01,   8.09607397e-02,
              7.08864391e-02,   1.70441616e-01,   7.66044653e+00]),
     array([  1.31287042e-11,   9.41946510e-12,   1.34416696e-11,
              4.12538302e-12,   2.88795995e-13,   3.32781615e-01,
              2.36923349e-01,   3.01392559e-01,   1.19670079e-01,
              9.23239851e-03,   2.51284631e+00,   5.82850181e+01,
              1.26094436e-11,   8.01165592e-12,   1.41308910e-11,
              3.03014003e-12,   1.54444934e-13,   2.08266789e-01,
              1.37399617e-01,   2.09628855e-01,   8.35185404e-02,
              7.10854224e-03,   1.46834801e-01,   7.95577542e+00]),
     array([  9.13603954e-12,   5.01873868e-12,   6.35464105e-12,
              2.95322721e-12,   6.71368833e-13,   3.34314715e-01,
              2.09079080e-01,   2.72375055e-01,   1.48334144e-01,
              3.58970051e-02,   2.31496982e+00,   5.76082908e+01,
              1.50560610e-11,   5.82286765e-12,   7.03186414e-12,
              1.84531767e-12,   5.78990200e-13,   2.01155056e-01,
              1.32081957e-01,   1.97385078e-01,   8.46523832e-02,
              3.04543908e-02,   1.77893254e-01,   7.77499170e+00]),
     array([  1.62439825e-12,   7.86366064e-13,   2.61628214e-12,
              1.29307418e-12,   4.17577643e-13,   2.27191406e-01,
              1.23456673e-01,   3.71994363e-01,   2.11882504e-01,
              6.54750539e-02,   1.85977155e+00,   5.34885835e+01,
              2.27414624e-12,   7.27513794e-13,   2.04848875e-12,
              7.09962107e-13,   4.56677801e-13,   1.53023343e-01,
              8.61699596e-02,   1.66756421e-01,   9.71067029e-02,
              4.62428555e-02,   2.67318472e-01,   6.01095592e+00]),
     array([  2.79926227e-12,   1.51051402e-12,   1.11832490e-11,
              2.25909712e-12,   8.29242348e-13,   1.66291754e-01,
              9.18012442e-02,   5.42671884e-01,   1.43516723e-01,
              5.57183953e-02,   2.22671557e+00,   6.00869966e+01,
              2.32834707e-12,   1.25748396e-12,   8.52141788e-12,
              1.11460490e-12,   5.08016212e-13,   1.21876505e-01,
              6.94251559e-02,   1.88383357e-01,   8.07994536e-02,
              4.47065147e-02,   1.83610138e-01,   4.87577561e+00]),
     array([  2.78941699e-12,   1.21015028e-12,   1.09108833e-12,
              1.27594138e-12,   4.24723414e-12,   2.45444381e-01,
              1.11965005e-01,   1.00956441e-01,   1.24902012e-01,
              4.16732160e-01,   2.02058621e+00,   5.54502355e+01,
              2.21276289e-12,   9.39475387e-13,   9.08649634e-13,
              5.73926555e-13,   1.51173448e-12,   1.27335071e-01,
              7.47410582e-02,   6.79958785e-02,   5.30310382e-02,
              1.38946617e-01,   1.99117981e-01,   6.22351953e+00]),
     array([  4.22452573e-12,   3.34871003e-12,   1.57803388e-12,
              7.59292313e-13,   3.28823481e-13,   3.93823737e-01,
              3.23679062e-01,   1.60762764e-01,   8.45808720e-02,
              3.71535649e-02,   2.29766683e+00,   5.32076258e+01,
              3.62407665e-12,   2.86438727e-12,   1.46214809e-12,
              6.87441160e-13,   3.44583827e-13,   1.89030587e-01,
              1.67396632e-01,   1.10387832e-01,   5.49515330e-02,
              3.32267992e-02,   1.98352959e-01,   7.12002322e+00])]



Generamos un pickle con los features en bruto.


```python
numpy_to_pickle(np.array(np_features_por_paciente, copy=True), "df_features_sin_estandarizar")
```

    Creando index...
    agrupacion_feature           media                                            \
    feature                      delta         theta         alpha          beta   
    tipo indice_paciente                                                           
    P    0                1.640200e-11  3.212565e-12  1.062605e-12  1.094488e-12   
         1                2.027273e-11  4.900092e-12  7.605571e-13  6.964696e-13   
         2                2.933076e-11  5.288517e-12  1.233875e-12  6.218898e-13   
         3                1.833007e-11  6.417983e-12  2.200516e-12  9.273606e-13   
         4                2.494467e-11  5.183651e-12  1.535496e-12  1.090633e-12   
         5                1.974450e-11  3.560001e-12  2.806103e-12  1.011536e-12   
         6                1.845960e-11  3.063844e-12  6.710482e-13  8.359896e-13   
         7                7.320086e-12  2.174221e-12  4.141556e-13  3.576999e-13   
         8                1.340707e-11  5.338362e-12  9.894577e-13  9.782136e-13   
         9                2.195274e-11  3.880261e-12  1.066262e-12  1.112371e-12   
    S    0                8.785838e-12  2.255187e-12  2.061689e-12  5.393903e-12   
         1                3.272551e-12  2.440789e-12  6.835303e-12  1.600645e-12   
         2                3.925761e-12  2.722708e-12  1.241359e-11  2.182036e-12   
         3                5.817862e-12  2.167575e-12  7.261025e-12  2.889613e-12   
         4                1.312870e-11  9.419465e-12  1.344167e-11  4.125383e-12   
         5                9.136040e-12  5.018739e-12  6.354641e-12  2.953227e-12   
         6                1.624398e-12  7.863661e-13  2.616282e-12  1.293074e-12   
         7                2.799262e-12  1.510514e-12  1.118325e-11  2.259097e-12   
         8                2.789417e-12  1.210150e-12  1.091088e-12  1.275941e-12   
         9                4.224526e-12  3.348710e-12  1.578034e-12  7.592923e-13   
    
    agrupacion_feature                                                             \
    feature                      gamma delta_norm theta_norm alpha_norm beta_norm   
    tipo indice_paciente                                                            
    P    0                2.441766e-13   0.638982   0.178471   0.081279  0.076343   
         1                3.017219e-13   0.688463   0.216744   0.038461  0.039018   
         2                1.247012e-13   0.747844   0.170894   0.049909  0.025969   
         3                9.505216e-14   0.585926   0.266792   0.096922  0.045377   
         4                4.406441e-13   0.684975   0.189109   0.061172  0.045592   
         5                7.724592e-14   0.650266   0.161245   0.129735  0.054537   
         6                3.766210e-13   0.720759   0.157007   0.043282  0.055533   
         7                1.569399e-13   0.617072   0.252727   0.055699  0.052216   
         8                1.830719e-13   0.590542   0.271264   0.062831  0.062313   
         9                2.144222e-13   0.706443   0.171452   0.054599  0.055954   
    S    0                6.346465e-12   0.321599   0.095496   0.089780  0.226482   
         1                1.130435e-13   0.226933   0.172814   0.458728  0.131343   
         2                1.746010e-13   0.210915   0.139672   0.515822  0.122847   
         3                1.948377e-12   0.265522   0.116801   0.382267  0.145388   
         4                2.887960e-13   0.332782   0.236923   0.301393  0.119670   
         5                6.713688e-13   0.334315   0.209079   0.272375  0.148334   
         6                4.175776e-13   0.227191   0.123457   0.371994  0.211883   
         7                8.292423e-13   0.166292   0.091801   0.542672  0.143517   
         8                4.247234e-12   0.245444   0.111965   0.100956  0.124902   
         9                3.288235e-13   0.393824   0.323679   0.160763  0.084581   
    
    agrupacion_feature                 ...               std                \
    feature              gamma_norm    ...             alpha          beta   
    tipo indice_paciente               ...                                   
    P    0                 0.024926    ...      1.138956e-12  1.459362e-12   
         1                 0.017314    ...      8.310352e-13  5.953013e-13   
         2                 0.005384    ...      9.175764e-13  2.963338e-13   
         3                 0.004982    ...      2.137412e-12  7.937704e-13   
         4                 0.019152    ...      1.069631e-12  5.481890e-13   
         5                 0.004217    ...      2.564946e-12  8.716742e-13   
         6                 0.023419    ...      4.877101e-13  6.371722e-13   
         7                 0.022285    ...      4.439258e-13  2.838362e-13   
         8                 0.013050    ...      8.621966e-13  8.735992e-13   
         9                 0.011553    ...      8.780474e-13  1.028570e-12   
    S    0                 0.266642    ...      1.898741e-12  4.212500e-12   
         1                 0.010182    ...      5.072665e-12  9.260869e-13   
         2                 0.010743    ...      1.131552e-11  1.307777e-12   
         3                 0.090022    ...      5.304215e-12  3.328995e-12   
         4                 0.009232    ...      1.413089e-11  3.030140e-12   
         5                 0.035897    ...      7.031864e-12  1.845318e-12   
         6                 0.065475    ...      2.048489e-12  7.099621e-13   
         7                 0.055718    ...      8.521418e-12  1.114605e-12   
         8                 0.416732    ...      9.086496e-13  5.739266e-13   
         9                 0.037154    ...      1.462148e-12  6.874412e-13   
    
    agrupacion_feature                                                             \
    feature                      gamma delta_norm theta_norm alpha_norm beta_norm   
    tipo indice_paciente                                                            
    P    0                2.964950e-13   0.192986   0.129180   0.081978  0.071857   
         1                2.466099e-13   0.197354   0.165659   0.037425  0.038939   
         2                5.875347e-14   0.152883   0.120009   0.048655  0.021731   
         3                1.021985e-13   0.217678   0.175358   0.088497  0.042723   
         4                2.804224e-13   0.185923   0.141365   0.053008  0.038079   
         5                4.998107e-14   0.198294   0.128831   0.109324  0.056322   
         6                6.014413e-13   0.169481   0.109362   0.043095  0.055817   
         7                2.045040e-13   0.216568   0.177382   0.053025  0.043276   
         8                1.168217e-13   0.199266   0.168754   0.055358  0.062098   
         9                2.196536e-13   0.182768   0.139415   0.051823  0.052629   
    S    0                4.430189e-12   0.174887   0.071391   0.068662  0.111956   
         1                8.423782e-14   0.161203   0.129342   0.196879  0.083956   
         2                1.010073e-13   0.153439   0.100725   0.205763  0.073636   
         3                2.923924e-12   0.166511   0.079560   0.176614  0.080961   
         4                1.544449e-13   0.208267   0.137400   0.209629  0.083519   
         5                5.789902e-13   0.201155   0.132082   0.197385  0.084652   
         6                4.566778e-13   0.153023   0.086170   0.166756  0.097107   
         7                5.080162e-13   0.121877   0.069425   0.188383  0.080799   
         8                1.511734e-12   0.127335   0.074741   0.067996  0.053031   
         9                3.445838e-13   0.189031   0.167397   0.110388  0.054952   
    
    agrupacion_feature                                    
    feature              gamma_norm     intra      inter  
    tipo indice_paciente                                  
    P    0                 0.032179  0.254985  10.348069  
         1                 0.018717  0.202654  11.173437  
         2                 0.004998  0.239314  12.378896  
         3                 0.005777  0.211214  11.184661  
         4                 0.019464  0.192878  11.467708  
         5                 0.003899  0.217435  11.604595  
         6                 0.035202  0.204486  10.164057  
         7                 0.025870  0.263664   8.971298  
         8                 0.012223  0.214571   9.768589  
         9                 0.013385  0.211949  11.972062  
    S    0                 0.120389  0.178850   7.926584  
         1                 0.012050  0.193726   6.396010  
         2                 0.008093  0.219512   6.594929  
         3                 0.070886  0.170442   7.660447  
         4                 0.007109  0.146835   7.955775  
         5                 0.030454  0.177893   7.774992  
         6                 0.046243  0.267318   6.010956  
         7                 0.044707  0.183610   4.875776  
         8                 0.138947  0.199118   6.223520  
         9                 0.033227  0.198353   7.120023  
    
    [20 rows x 24 columns]


Generamos también ahora un pickle con las/los features pero ahora estandarizados.


```python
a_np_copy = np.array(np_features_por_paciente, copy=True)

scaler = preprocessing.StandardScaler().fit(a_np_copy)
#print (scaler)
np_features_por_paciente_norm = scaler.transform(a_np_copy)

print ("media: ")
print(scaler.mean_)
print ("std: ")
print(scaler.var_)
print ("samples: {}".format(scaler.n_samples_seen_))

#scalerDelNorm = preprocessing.StandardScaler().fit(np_features_por_paciente_norm)
## Esto deberia devolver un vector de medias con 0s y uno de desvios con 1s, pero la media no los da.
#print ("media: ")
#print(scalerDelNorm.mean_)
#print ("std: ")
#print(scalerDelNorm.var_)

#unPasoMasScalerDelNorm = preprocessing.StandardScaler().fit(scalerDelNorm.transform(np_features_por_paciente_norm))
## Esto deberia devolver un vector de medias con 0s y uno de desvios con 1s
#print ("media: ")
#print(unPasoMasscalerDelNorm.mean_)
#print ("std: ")
#print(unPasoMasscalerDelNorm.var_)

#df_features_normalizado_compress = pd.DataFrame({
#    "Capacidad cognitiva": (["Reducida"] * (N_P * 24)) + (["Normal"] * (N_S * 24)),
#    "Features": list(itertools.chain(*[list(give_expected_ordered_keys()) for _ in range(N_P+N_S)])),
#    "Valores": list(itertools.chain(*[list(paciente) for paciente in np_features_por_paciente_norm]))
#})
```

    media: 
    [  1.22834290e-11   3.69498494e-12   3.87883212e-12   1.67294314e-12
       8.79006301e-13   4.67804431e-01   1.82869705e-01   1.93531906e-01
       9.85899216e-02   5.72040351e-02   2.25498518e+00   5.42195480e+01
       1.27711045e-11   3.49634631e-12   3.45130198e-12   1.25622801e-12
       6.63534335e-13   1.78496478e-01   1.25177323e-01   1.10532130e-01
       6.44018789e-02   3.41909793e-02   2.07440365e-01   8.87861917e+00]
    std: 
    [  6.77913851e-23   4.01460556e-24   1.67288690e-23   1.58382884e-24
       2.44143192e-24   4.17589667e-02   3.83029150e-03   2.83991607e-02
       3.14527263e-03   1.00247485e-02   1.99600977e-02   9.09067401e+00
       6.19814255e-23   3.25129023e-24   1.46628709e-23   1.07692142e-24
       1.16633255e-24   7.12535638e-04   1.21472262e-03   3.95851355e-03
       4.87637468e-04   1.29745886e-03   9.11895875e-04   5.06894719e+00]
    samples: 20



```python
#print (df_features_normalizado_compress)
printear_comparacion_sin_y_con_estandarizar(np_features_por_paciente, np_features_por_paciente_norm, give_expected_ordered_keys())
```


![png](readme_images/output_21_0.png)



![png](readme_images/output_21_1.png)



![png](readme_images/output_21_2.png)



![png](readme_images/output_21_3.png)



![png](readme_images/output_21_4.png)



![png](readme_images/output_21_5.png)



![png](readme_images/output_21_6.png)



![png](readme_images/output_21_7.png)



![png](readme_images/output_21_8.png)



![png](readme_images/output_21_9.png)



![png](readme_images/output_21_10.png)



![png](readme_images/output_21_11.png)



![png](readme_images/output_21_12.png)



![png](readme_images/output_21_13.png)



![png](readme_images/output_21_14.png)



![png](readme_images/output_21_15.png)



![png](readme_images/output_21_16.png)



![png](readme_images/output_21_17.png)



![png](readme_images/output_21_18.png)



![png](readme_images/output_21_19.png)



![png](readme_images/output_21_20.png)



![png](readme_images/output_21_21.png)



![png](readme_images/output_21_22.png)



![png](readme_images/output_21_23.png)



```python
numpy_to_pickle(np_features_por_paciente_norm, "df_features_estandarizado")
```

    Creando index...
    agrupacion_feature       media                                          \
    feature                  delta     theta     alpha      beta     gamma   
    tipo indice_paciente                                                     
    P    0                0.500218 -0.240771 -0.688548 -0.459638 -0.406289   
         1                0.970334  0.601456 -0.762397 -0.775901 -0.369460   
         2                2.070471  0.795315 -0.646674 -0.835162 -0.482753   
         3                0.734391  1.359020 -0.410337 -0.592436 -0.501728   
         4                1.537761  0.742978 -0.572930 -0.462701 -0.280550   
         5                0.906179 -0.067369 -0.262275 -0.525551 -0.513124   
         6                0.750122 -0.314996 -0.784281 -0.665039 -0.321525   
         7               -0.602819 -0.758998 -0.847090 -1.045086 -0.462120   
         8                0.136471  0.820192 -0.706432 -0.552029 -0.445396   
         9                1.174379  0.092469 -0.687654 -0.445428 -0.425332   
    S    0               -0.424797 -0.718588 -0.444279  2.956657  3.499155   
         1               -1.094409 -0.625956  0.722837 -0.057448 -0.490214   
         2               -1.015074 -0.485254  2.086689  0.404523 -0.450817   
         3               -0.785271 -0.762314  0.826923  0.966760  0.684394   
         4                0.102662  2.857029  2.338048  1.948697 -0.377733   
         5               -0.382264  0.660672  0.605318  1.017307 -0.132887   
         6               -1.294585 -1.451662 -0.308685 -0.301842 -0.295313   
         7               -1.151892 -1.090247  1.785880  0.465755 -0.031849   
         8               -1.153088 -1.240155 -0.681584 -0.315456  2.155654   
         9               -0.978788 -0.172822 -0.562529 -0.725982 -0.352115   
    
    agrupacion_feature                                                          \
    feature              delta_norm theta_norm alpha_norm beta_norm gamma_norm   
    tipo indice_paciente                                                         
    P    0                 0.837666  -0.071074  -0.666111 -0.396677  -0.322384   
         1                 1.079809   0.547332  -0.920190 -1.062210  -0.398412   
         2                 1.370391  -0.193502  -0.852261 -1.294888  -0.517556   
         3                 0.578035   1.356012  -0.573282 -0.948825  -0.521571   
         4                 1.062737   0.100820  -0.785426 -0.944991  -0.380050   
         5                 0.892889  -0.349412  -0.378571 -0.785505  -0.529213   
         6                 1.237849  -0.417890  -0.891581 -0.767738  -0.337433   
         7                 0.730451   1.128753  -0.817898 -0.826884  -0.348758   
         8                 0.600624   1.428262  -0.775578 -0.646849  -0.440994   
         9                 1.167791  -0.184488  -0.824430 -0.760237  -0.455946   
    S    0                -0.715465  -1.411769  -0.615662  2.280417   2.091798   
         1                -1.178718  -0.162472   1.573673  0.584005  -0.469644   
         2                -1.257104  -0.697976   1.912467  0.432528  -0.464033   
         3                -0.989883  -1.067524   1.119954  0.834446   0.327774   
         4                -0.660743   0.873392   0.640045  0.375876  -0.479124   
         5                -0.653240   0.423488   0.467855  0.886979  -0.212807   
         6                -1.177455  -0.959988   1.058995  2.020097   0.082608   
         7                -1.475471  -1.471472   2.071795  0.801081  -0.014838   
         8                -1.088133  -1.145669  -0.549342  0.469166   3.590841   
         9                -0.362029   2.275179  -0.194452 -0.249793  -0.200257   
    
    agrupacion_feature      ...          std                                 \
    feature                 ...        alpha      beta     gamma delta_norm   
    tipo indice_paciente    ...                                               
    P    0                  ...    -0.603870  0.195745 -0.339861   0.542809   
         1                  ...    -0.684283 -0.636885 -0.386052   0.706450   
         2                  ...    -0.661683 -0.924978 -0.559998  -0.959542   
         3                  ...    -0.343123 -0.445636 -0.519770   1.467841   
         4                  ...    -0.621974 -0.682284 -0.354744   0.278225   
         5                  ...    -0.231472 -0.370566 -0.568121   0.741657   
         6                  ...    -0.773943 -0.596538 -0.057495  -0.337742   
         7                  ...    -0.785377 -0.937021 -0.425040   1.426262   
         8                  ...    -0.676146 -0.368711 -0.506230   0.778088   
         9                  ...    -0.672006 -0.219376 -0.411013   0.160035   
    S    0                  ...    -0.405452  2.848737  3.487744  -0.135212   
         1                  ...     0.423419 -0.318132 -0.536401  -0.647850   
         2                  ...     2.053743  0.049673 -0.520873  -0.938721   
         3                  ...     0.483889  1.997370  2.093014  -0.449005   
         4                  ...     2.788978  1.709386 -0.471393   1.115270   
         5                  ...     0.935065  0.567661 -0.078284   0.848847   
         6                  ...    -0.366345 -0.526395 -0.191539  -0.954287   
         7                  ...     1.324062 -0.136472 -0.144002  -2.121125   
         8                  ...    -0.664014 -0.657483  0.785393  -1.916634   
         9                  ...    -0.519468 -0.548097 -0.295333   0.394634   
    
    agrupacion_feature                                                         \
    feature              theta_norm alpha_norm beta_norm gamma_norm     intra   
    tipo indice_paciente                                                        
    P    0                 0.114840  -0.453845  0.337590  -0.055859  1.574465   
         1                 1.161500  -1.161972 -1.153092  -0.429586 -0.158501   
         2                -0.148300  -0.983480 -1.932346  -0.810451  1.055501   
         3                 1.439780  -0.350221 -0.981742  -0.788827  0.124952   
         4                 0.464447  -0.914285 -1.192041  -0.408850 -0.482241   
         5                 0.104833  -0.019199 -0.365904  -0.840978  0.330971   
         6                -0.453787  -1.071846 -0.388773   0.028057 -0.097829   
         7                 1.497864  -0.914026 -0.956674  -0.231002  1.861856   
         8                 1.250308  -0.876940 -0.104337  -0.609874  0.236147   
         9                 0.408511  -0.933130 -0.533133  -0.577609  0.149306   
    S    0                -1.543234  -0.665489  2.153488   2.393049 -0.946790   
         1                 0.119493   1.372396  0.885514  -0.614677 -0.454166   
         2                -0.701597   1.513603  0.418149  -0.724531  0.399771   
         3                -1.308844   1.050311  0.749863   1.018745 -1.225221   
         4                 0.350683   1.575046  0.865692  -0.751867 -2.006965   
         5                 0.198108   1.380443  0.917038  -0.103736 -0.978458   
         6                -1.119201   0.893630  1.481028   0.334586  1.982875   
         7                -1.599643   1.237369  0.742559   0.291934 -0.789143   
         8                -1.447119  -0.676072 -0.514925   2.908242 -0.275597   
         9                 1.211358  -0.002293 -0.427956  -0.026768 -0.300931   
    
    agrupacion_feature              
    feature                  inter  
    tipo indice_paciente            
    P    0                0.652673  
         1                1.019270  
         2                1.554689  
         3                1.024255  
         4                1.149974  
         5                1.210774  
         6                0.570942  
         7                0.041165  
         8                0.395291  
         9                1.373989  
    S    0               -0.422858  
         1               -1.102680  
         2               -1.014328  
         3               -0.541066  
         4               -0.409892  
         5               -0.490189  
         6               -1.273706  
         7               -1.777910  
         8               -1.179294  
         9               -0.781101  
    
    [20 rows x 24 columns]


<blockquote>
<p>Aunque visualmente no lo parezca, el desvío quedó en 1 y la medía más cerca del 0 (aunque no en 0). Algo a tener en cuenta es que estamos estandarizando por <i>feature</i>, y eso incluye tanto a los pacientes P como a los S.  
La apertura de los valores tiene sólo sentido viendo que los valores antes de estandarizar son muy pequeños.</p>
</blockquote>

<h4>Resumen de datos sin estandarizar</h4>


```python
print(np.array([p[0] for p in np_features_por_paciente]))
print("media {}".format(np.mean(np.array([p[0] for p in np_features_por_paciente]))))
print("std {}".format(np.std(np.array([p[0] for p in np_features_por_paciente]))))
```

    [  1.64020028e-11   2.02727252e-11   2.93307575e-11   1.83300729e-11
       2.49446658e-11   1.97445003e-11   1.84595979e-11   7.32008586e-12
       1.34070686e-11   2.19527438e-11   8.78583783e-12   3.27255133e-12
       3.92576050e-12   5.81786189e-12   1.31287042e-11   9.13603954e-12
       1.62439825e-12   2.79926227e-12   2.78941699e-12   4.22452573e-12]
    media 1.2283428962309579e-11
    std 8.23355239968603e-12


<h4>Resumen de datos estandarizando</h4>


```python
print(np.array([p[0] for p in np_features_por_paciente_norm]))
print("media {}".format(np.mean(np.array([p[0] for p in np_features_por_paciente_norm]))))
print("std {}".format(np.std(np.array([p[0] for p in np_features_por_paciente_norm]))))
```

    [ 0.50021833  0.97033405  2.07047064  0.73439066  1.53776112  0.90617889
      0.75012202 -0.60281916  0.13647082  1.17437947 -0.42479734 -1.09440946
     -1.01507442 -0.78527065  0.10266228 -0.38226385 -1.29458467 -1.15189243
     -1.15308818 -0.97878811]
    media -1.27675647831893e-16
    std 1.0

