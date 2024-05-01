# TC3002B_MvsW_A01703259
Ricardo Nuñez Alanis
## Dataset
En el siguiente link se encuntra la descarga del dataset: https://www.kaggle.com/datasets/hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images
Dentro de este dataset se encuentran 3 clases de 5000 imagenes cada una; Estas clases son: Zapatos, Sandalias y Botas con un tamaño de resolución de
136 x 102 cada una de estas
### Preprocesamiento
Se separo unicamente en train y test con un 75% (3750) de estas imagenes en train y en test el 25% (1250) restante.
Para que las imagenes puedan ser leidas por el modelo y que los valores de los pixeles sean entre [0,1] se uso la función
resxale para lograr esto.

## Modelo
Es un modelo de clasificación secuencial de categorial multiples (3) en el cual se esta usando un input_shape de 150x150, con
2 capas de convolutivas de 16 y 32 con un kernel de 3x3, usamos maxpooling2D para reducir las dimensiones de los datos y se
continua en un flatten para tener en un solo vector todos los datos ya procesados y al final manejamos unas capas densas, para
manejar unas neuronas que permiten la conexión entre distintas capas y por último usamos la función de dropout para que nuestro
modelo no sufra de overfitting.

## Resultados
Dentro de los archivos .ipynb se encuntran los resultados de este modelo.
