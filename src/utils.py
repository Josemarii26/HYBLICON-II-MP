import os
import random
import string

# Función que genera un identificador aleatorio de una longitud específica
def generate_random_id(length=5):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# Función que registra los resultados de una predicción en un archivo de log
def log_prediction(log_file_path, request_id, timestamp, lift_id, model_version, special_objects, wheelchairs, walkers, crutches, num_people, gender_classification):

    """
    Registra en un archivo CSV la información de una predicción de un sistema de visión artificial.
    
    Args:
        log_file_path (str): Ruta del archivo de log.
        request_id (str): Identificador único de la solicitud.
        timestamp (str): Fecha y hora de la solicitud.
        lift_id (str): Identificador del ascensor o dispositivo.
        model_version (str): Versión del modelo utilizado para la predicción.
        special_objects (int): Número de objetos especiales detectados.
        wheelchairs (int): Número de sillas de ruedas detectadas.
        walkers (int): Número de andadores detectados.
        crutches (int): Número de muletas detectadas.
        num_people (int): Número de personas detectadas.
        gender_classification (str): Clasificación de género y edad  detectada.
    
    """
    
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        with open(log_file_path, 'a') as log_file:
            log_file.write("request_id,timestamp,lift_id,model_version,special_objects,wheelchairs,walkers,crutches,num_people,gender_classification\n")
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"{request_id},{timestamp},{lift_id},{model_version},{special_objects},{wheelchairs},{walkers},{crutches},{num_people},{gender_classification}\n")
