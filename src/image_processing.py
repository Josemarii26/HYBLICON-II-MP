import cv2
import numpy as np

# Función que determina si la imagen está borrosa usando la varianza de Laplacian
def es_borrosa(gris, umbral=40):
    varianza = cv2.Laplacian(gris, cv2.CV_64F).var()
    return varianza < umbral

# Función que determina si la imagen tiene franjas unicolor
def tiene_franjas_unicolor(gris, tamaño_bloque=160, umbral=5):
    # Tamaño de la imagen
    alto, ancho = gris.shape

    # Iterar sobre bloques
    for y in range(0, alto, tamaño_bloque):
        for x in range(0, ancho, tamaño_bloque):
            # Limitar los bloques al tamaño de la imagen
            bloque = gris[y:min(y+tamaño_bloque, alto), x:min(x+tamaño_bloque, ancho)]
            
            # Calcula la varianza del bloque
            varianza_bloque = np.var(bloque)

            # Si la varianza del bloque es menor que el umbral, es una región unicolor
            if varianza_bloque < umbral:
                return True  # Se ha detectado una franja o región unicolor

    return False  # No se detectó ninguna región unicolor

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
