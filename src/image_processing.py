import cv2
import numpy as np

# Función que determina si la imagen está borrosa usando la varianza de Laplacian
def es_borrosa(gris, umbral=5):
    varianza = cv2.Laplacian(gris, cv2.CV_64F).var()
    return varianza < umbral

# Función que determina si la imagen tiene franjas unicolor
def tiene_franjas_unicolor(gris, tamaño_bloque=200, umbral=5):
    alto, ancho = gris.shape
    return any(np.var(gris[y:min(y+tamaño_bloque, alto), x:min(x+tamaño_bloque, ancho)]) < umbral     # Evaluar la varianza en cada bloque, si algún bloque tiene varianza baja, se considera franja unicolor
               for y in range(0, alto, tamaño_bloque) for x in range(0, ancho, tamaño_bloque))

# Función para evaluar la calidad de la imagen
def evaluar_calidad(imagen):
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    if tiene_franjas_unicolor(gris):
        return "Imagen con franjas unicolor"
    if es_borrosa(gris):
        return "Imagen borrosa"
    return "Calidad suficiente"

# Función para redimensionar la imagen manteniendo la relación de aspecto
def redimensionar_imagen(imagen, ancho_max=800):
    alto, ancho = imagen.shape[:2]
    if ancho > ancho_max:
        return cv2.resize(imagen, (ancho_max, int(alto * (ancho_max / ancho))))
    return imagen
