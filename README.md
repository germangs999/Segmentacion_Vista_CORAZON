# Segmentacion_Vista_CORAZON
Segmentación de 5 zonas del corazón: ventrículo izquierdo y derecho, miocardio izquierdo, y atrio izquierdo y derecho. Además, clasificación de la vista del volumen.

Las librerías que se utilizarán para este proyecto son:

```python
##Programa para clasificar la vista de un volumen e imgs de corazon
import os
import numpy as np
import skimage.io as io
import cv2
import tensorflow as tf
from optparse import OptionParser
import sys
sys.path.append('./funcs/')
sys.path.append('./nets/')
from fun_vista_segmentacion import clasificacion_vista, segmentacion_4c, segmentacion_a4c_vista
```
Las funciones de clasificación y segmentación están incluidas en **fun_vista_segmentacion.py**.

El volumen de prueba es **patient0002_4CH_sequence.mhd** (cuyos datos están asociados al archivo RAW) y se lee utilizando **SimpleITK**:

```python
nombre = 'patient0002_4CH_sequence.mhd'
vol = io.imread(nombre, plugin='simpleitk')
```

Para este ejemplo brindamos tres funciones: clasificación de la vista, segmentación de las 5 regiones del corazón y clasificación y segmentación al mismo tiempo.

La clasificación de vista se obtiene con la siguiente función cuya entrada es el volumen (video) y la salida es la etiqueta del tipo de vista (string):
```python
vista = clasificacion_vista(vol)
print(vista)
```
La segmentación de las cavidades se realiza con la función **segmentacion_4c** y cabe recordar que **esta función realiza forzosamente la segmentación en volúmenes de 4 cámaras independientemente de la etiqueta generada por la clasificación de vista**.

```python
#Segmentacion de las cinco cavidades en 4 cámaras
# segmentacion = 1 - ventriculo izquierdo, 2 - atrio izquierdo, 3-miocardio izquierdo
# 4 - ventriculo derecho 5 - atrio derecho
vol_segmentado = segmentacion_4c(vol)
```
cuya entrada es el mismo volumen y la salida es la segmentación de las cinco regiones:

1. Ventrículo izquierdo
2. Atrio izquierdo
3. Miocardio izquierdo
4. Ventrículo derecho
5. Atrio derecho

Y las etiquetas se pueden observar de la siguiente manera:


Finalmente, la función **segmentacion_a4c_vista** realiza ambas tareas.

```python
vol_segmentado1, vista1 = segmentacion_a4c_vista(vol)
print(vista1)
```
Los archivos **vistas.txt**, **util.py** y **fun_vista_segmentacion.py** son necesarias para el correcto funcionamiento del programa.


