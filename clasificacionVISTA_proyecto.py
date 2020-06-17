# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 22:32:26 2020

@author: germa
"""
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

import warnings
warnings.filterwarnings('ignore')

#Leer el volumen
nombre = 'D:/germa/DriveUP/Github/EtiquetaSegmentacionCorazon/patient0002_4CH_sequence.mhd'
vol = io.imread(nombre, plugin='simpleitk')

#Clasificacion de la vista
vista = clasificacion_vista(vol)
print(vista)

#Segmentacion de las cinco cavidades en 4 cámaras
# segmentacion = 1 - ventriculo izquierdo, 2 - atrio izquierdo, 3-miocardio izquierdo
# 4 - ventriculo derecho 5 - atrio derecho
vol_segmentado = segmentacion_4c(vol)

# Obtener segmentacion y vista (no necesariamente corresponde a a4c, cuatro camaras)
vol_segmentado1, vista1 = segmentacion_a4c_vista(vol)
print(vista1)
