# Técnicas de Machine Learning
# 1. Introducción

En este trabajo práctico vamos a utilizar técnicas de machine learning univariadas (una variable/feature) y multivariadas (mutiples variabes/features).

Algunas definiciones:
- *Feature*: En Machine Learning, un feature es una propiedad o característica individual y cuantificable del dominio de nuestro problema.
- *samples* o muestra: Se refiere a cada muestra observable. En este caso, una muestra es un sujeto.
- *label* o etiqueta: Cuando se realiza una clasificación, cada muestra tiene asociado un label.

## Clasificación

En Machine Learning, el problema de Clasificación se define como la identificación del grupo o categoría al que pertenences una muestra. El ejeplo mas utilizado en la literatura es el de clasificar un correo electrónico como *Spam* o no.

Cuando la clasificación es supervisada, el modelo se entrena con un set de datos de ejemplo. Es decir, para el clasificador de spam, existe un conjunto de emails donde un supervisor ya clasificó manualmente y le puso le etiqueta correspondiente a cada uno.

Los algoritmos de clasificación trabajan sobre features. Para cada muestra, se deben extraer uno o mas features que junto con las etiquetas de cada muestra, consisten en el set de datos que el clasificador utiliza para aprender. En el caso del correo electrónico, ejemplos pueden ser el dominio de la dirección de envío, si esta se encuentra en la lista de direcciones conocidas del usuario, el idioma del mensaje, la longitud, la cantidad de archivos adjuntos, (etc).

Luego de entrenar el modelo, se pueden utilizar para predecir la etiqueta de nuevas muestras. Si contamos con las etiquetas de las nuevas muestras (mediante un supervisor), podemos estimar medidas de performance del clasificador.

Para mas información acerca de clasificadores supervisados: [Scikit Learn](http://scikit-learn.org/stable/supervised_learning.html)


- [Wikipedia](https://en.wikipedia.org/wiki/Statistical_classification)

## Validación Cruzada *(Cross Validation)*

Idealmente, al cuantificar la performance de un clasificador, este se entrena en un set de datos, comunmente denominado *training set* y luego se mide la performance en otro set de datos independiente, llamado *testing set*. Si al finalizar la cuantificación, uno decide cambiar el modelo o los parámetros del clasificador, se debe obtener un nuevo *testing set* dado que los *labels* del set de prueba ya fueron observados y utilizados.

Dependiendo del dominio del problema, obtener un nuevo set de prueba puede ser sencillo (por ejemplo, utilizando posts de facebook o tweets) o muy costoso (como el de este trabajo práctico). En casos donde la cantidad de muestras es acotada y dificil de obtener, se utiliza la técnica de *cross validation* o validación cruzada (CV). El método consiste en dividir el set de datos en dos (*train* y *test*) y estimar la performance utilizando diferentes divisiones del mismo. Existen distintas mecanismos específicos, cada uno representando distintas caracteristicas de las muestras disponibles.

* *K-Fold*: Las muestras se dividen en ![](https://latex.codecogs.com/gif.latex?\inline&space;K) grupos (*folds*) de igual tamaño (si es posible). Se entrena en ![](https://latex.codecogs.com/gif.latex?\inline&space;K-1) folds y se prueba en el fold restante. Al final se obtienen ![](https://latex.codecogs.com/gif.latex?\inline&space;K) estimaciones de performance. El valor a reportar es la media.
* *Repeated K-Fold*: Idem al anterior, pero realizando varias repeticiones, obteniendo folds distintos en cada repetición.
* Leave one out: Cada set de entrenamiento se crea tomando ![](https://latex.codecogs.com/gif.latex?\inline&space;N-1) de las ![](https://latex.codecogs.com/gif.latex?\inline&space;N) muestras. El set de prueba consiste en la muestra que quedó fuera. Al final, se obtienen ![](https://latex.codecogs.com/gif.latex?\inline&space;N) estimaciones. No es posible realizar repeticiones debido a que no hay ningun factor aleatorio asociado.
* Shuffle Split: Genera los set de entreamiento y test realizando una división aleatoria segun los tamaños definidos por el usuario. Se realizan tantas repeticiones como indica el parámetro *n_splits*.

Un problema que puede estar asociado a las muestras es que el set de datos puede no estar balanceado. Es decir, para dos etiquetas A y B, podemos tener un 80% de muestras A y un 20 % de muestras B. Por lo tanto, es deseable utilizar sets de CV que respeten esta distribución de clases. Los mecanismos *Stratified* realizan las divisiones teniendo en cuenta las distrubiciones de las clases.

* Stratified K-Fold
* Stratified Shuffle Split

Todos estos métodos se encuentran implementados en [Scikit Learn - Model Selection](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection)

- [Wikipedia](https://en.wikipedia.org/wiki/Cross-validation_(statistics))
- [Scikit Learn](http://scikit-learn.org/stable/modules/cross_validation.html)

## Curva ROC
Una curva _Receiver Operating Characteristic_ (ROC) es un gráfico que muestra la capacidad de clasificación de una variable o clasificador con respecto a dos clases. Ilustra como varia la performance según el _threshold_ elegido. Para cada threshold elegido (eje X), ilustra el radio de _True Positives_ (TPR) vs el radio de _False Positives_ (FPR).

Dado un feature, se puede computar la curva ROC para graficar la capacidad de discriminación de ese feature con respecto a la pertenencia al grupo S o P. La curva ROC se cuantifica midiendo el area bajo la curva (AUC).

Para mas información:
- [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- [Kennis Research](https://kennis-research.shinyapps.io/ROC-Curves/)
- [Scikit Learn](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)

## Preprocesamiento

Muchos de los algoritmos de clasificación requieren que las variables tengan una distribución similar a una Gaussiana con media igual a cero y varianza unitaria. Para ellos, suele ser importante estandarizar los features.

Es común utilizar un *Standard Scaler* que computa la media y desviación estandard en el set de entrenamiento, para luego aplicarse a todas las muestras.

Si nuestros datos pueden tener outliers, el *Robust Scaler* realiza una estimación mas robusta de la media y rango de los datos.

- [Scikit Learn](http://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)

## *Feature Selection*

Los metodos de selección de features se utilizan para identificar features innecesarios, irrelevantes y/o redundantes que no contribuyen a la predicción e incluso pueden decrementar la performance del clasificador.

En nuestro ejemplo, contamos con 20 muestras (10 de cada grupo). Si la cantidad de features que utilizamos es muy grande, incrementa la complejidad del modelo, pero tambien incrementa la cantidad de datos que necesitamos utilizar para entrenar. Una *thumb rule* que suele funcionar: ![](https://latex.codecogs.com/gif.latex?\inline&space;N>2^d) donde *N* es el número de muestras y *d* es el número de features.

Entre los métodos univariados soportados por Scikit Learn se encuentran:
* *SelectKBest*: Elije los K mejores (K definido por el usuario).
* *SelectPercentile*: Elije los K% mejores basados en la cantidad total de features (percentil definido por el usuario).

- [Scikit Learn](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_selection)

# 2. Ejercicios

Para realizar los ejercicios vamos a utiliar los datos del trabajo práctico número 2. Cada sujeto es un *sample*, con su correspondiente *label* S o P.

Los features que vamos a utilizar son:
* Potencia para cada banda de frecuencia (Delta, Theta, Alpha, Beta y Gamma)
* Potencia normalizada para las mismas bandas de frecuencia.
* Una medida de información intra-electrodo (a elección)
* Una medida de informacion inter-electrodo (a elección)

La normalización a la que se refiere el enunciado del [TP2](http://www.dc.uba.ar/materias/cienciadatos/tps/tp2/enunciado) corresponde a dividir el poder de una banda de frecuencia por la suma del poder total:
![Normalizacion](https://latex.codecogs.com/gif.latex?%5Cinline%20%5Cleft%20%5C%7C%20x%20%5Cright%20%5C%7C%20%3D%20%5Cfrac%7Bx%7D%7B%5Cdelta&plus;%5Ctheta&plus;%5Calpha&plus;%5Cbeta&plus;%5Cgamma%7D)

Cada marcador especificado en el [TP2](http://www.dc.uba.ar/materias/cienciadatos/tps/tp2/enunciado) se debe computar por epochs. Es decir, **para cada epoch, debemos obtener un valor asociado al marcador** (incluyendo la normalización en el caso de los espectrales). Para computar el feature, se pueden relizar computando la media entre epochs (promedio en el tiempo) o la desviación estándard entre epochs (fluctuaciones en el tiempo). Tradicionalmente, la media suele ser la operación elégida para cuantificar los features cuando se tienen varias medidas en el tiempo. Sin embargo, no es cierto que sea la mas representativa en todos los casos. En el dominio de este trabajo práctico, dependiendo del feature, las fluctuaciones en el tiempo pueden ser incluso más importantes que el valor medio.

En total, para cada muestra debemos obtener 24 features: 5 espectrales, 5 espectrales normalizadas, 1 intra-electrodo y 1 inter-electrodo, computadas como media y desviación estandar entre epochs.


## 2.1 Análisis Univariado

a) Para cada feature, computar la curva ROC y graficarla como en el ejemplo:
![Curva ROC](https://gist.githubusercontent.com/fraimondo/d217a0736db06afa2ea0b1c202e37c8c/raw/e137da57c2ddd13e121e0a02076fd5dd430349a9/figura_2_1_d.png)

*Curva ROC para alpha*

b) Utilizando una técnica de cross validación, estimar la performance de un classificador *Logistic Regression* para cada feature y graficar la curva ROC correspondiente. ¿Cuál es su conclución respecto a los resultados obtenidos en el punto anterior?

## 2.2 Análisis Multivariado
a) Utilizar todos los features y entrenar un clasificador basado en Support Vector Machine. Computar la curva roc y graficarla. No olvidar reportar el area bajo la curva.

b) Repetir el punto a), pero utilizando un [*pipeline*](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) de Scikit-Learn con los siguientes 3 pasos:

1. *Standard Scaler*
2. *Feature Selection* utilizando solo el 10%.
3. *SVC*

¿Qué diferencia encuentra? ¿Y si utilizamos el 20% de los features? ¿Qué pasa si probamos y encontramos que utilizando el 35% de los features obtenemos la mejor AUC?