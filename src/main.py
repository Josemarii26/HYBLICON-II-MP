import cv2
from ultralytics import YOLO
from datetime import datetime
from utils import generate_random_id, log_prediction
from image_processing import evaluar_calidad, redimensionar_imagen
from model_analysis import analyze_bboxes

# Función principal para procesar una imagen y realizar predicciones
def predict_image(image_path, log_file_path, lift_id, model_version):
    """
    Esta función carga una imagen, ejecuta modelos de detección y clasificación sobre la misma,
    y guarda los resultados en un archivo de log. Además, dibuja líneas de referencia y cajas sobre la imagen.

    """

    # Cargar los modelos necesarios
    detection_model_person = YOLO("./models/bestY11-pose.pt")  # Modelo de detección de personas
    classification_model = YOLO('./models/bestClasiNew2.pt')  # Modelo de clasificación de género
    detection_model_mobility = YOLO('./models/bestY9-v8.pt')  # Modelo de detección de objetos de movilidad

    # Generar un ID único y obtener la marca de tiempo actual
    request_id = generate_random_id()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Cargar la imagen desde la ruta proporcionada
    image = cv2.imread(image_path)
    if image is None:
        print("Error: No se pudo cargar la imagen.")
        return

    # Redimensionar la imagen y evaluar su calidad
    image_redimensionada = redimensionar_imagen(image)
    calidad = evaluar_calidad(image_redimensionada)
    print(f"Resultado de la evaluación de calidad: {calidad}")
    
    if calidad != "Calidad suficiente":
        print(f"Imagen no apta para análisis: {calidad}")
        return

    # Realizar predicciones de personas en la imagen usando el modelo de detección
    preds_person = detection_model_person.predict(image_redimensionada, conf=0.3)[0]
    num_people = len(preds_person.boxes)  # Número de personas detectadas
    final_image = preds_person.plot()  # Imagen con las predicciones dibujadas

    gender_classification = []  # Lista para almacenar las clasificaciones de género

    # Verificar si se detectaron personas
    if num_people > 0:
        # Obtener dimensiones de la imagen
        height, width = image_redimensionada.shape[:2]  
        
        # Definir límites para las zonas de interés en la imagen (1/3 y 2/3 de la altura)
        lower_limit_1_3 = height / 3      
        upper_limit_1_3 = (2 * height) / 3  
        lower_limit_2_3 = upper_limit_1_3  
        upper_limit_2_3 = height              

        print("Se detectan personas en la imagen.")

        # Dibujar líneas horizontales que dividen la imagen en tercios
        cv2.line(final_image, (0, int(lower_limit_1_3)), (width, int(lower_limit_1_3)), (0, 0, 255), 2)  # Línea a 1/3 de altura (Rojo)
        cv2.line(final_image, (0, int(upper_limit_1_3)), (width, int(upper_limit_1_3)), (0, 0, 255), 2)  # Línea a 2/3 de altura (Rojo)


        # Analizar puntos clave y realizar clasificación de género
        for i, result in enumerate(preds_person):
            keypoints = result.keypoints  # Acceder a los puntos clave de la predicción
            
            if keypoints is None:
                print(f"No se detectaron puntos clave para la persona {i}")
                continue

            points_in_zone_1_3 = []  # Puntos clave en la zona 1/3 a 2/3
            points_in_zone_2_3 = []  # Puntos clave en la zona 2/3 a 3/3
            left_eye_detected = False
            right_eye_detected = False

            # Verificar si se detectaron ojos y puntos clave en las zonas de interés
            for person_keypoints in keypoints.xy:
                left_eye = person_keypoints[1][1]  # Coordenada del ojo izquierdo
                right_eye = person_keypoints[2][1]  # Coordenada del ojo derecho
                
                if left_eye != 0:  # Verificar si el ojo izquierdo está presente
                    left_eye_detected = True
                if right_eye != 0:  # Verificar si el ojo derecho está presente
                    right_eye_detected = True

                # Verificar en qué zonas están los puntos clave
                for idx in [0, 1, 2, 3, 4, 11, 12]:  # Índices de puntos clave relevantes
                    y_value = person_keypoints[idx][1]
                    if y_value != 0:  # Ignorar puntos que son cero
                        if lower_limit_1_3 <= y_value <= upper_limit_1_3:
                            points_in_zone_1_3.append(idx + 1)
                        if lower_limit_2_3 <= y_value <= upper_limit_2_3:
                            points_in_zone_2_3.append(idx + 1)

            # Clasificar si se detectaron ambos ojos y los puntos clave están en las zonas de interés
            if left_eye_detected and right_eye_detected and (points_in_zone_1_3 or points_in_zone_2_3):
                print(f"Iniciando clasificación de género y edad para la persona {i}.")
                gender_classification.extend(analyze_bboxes(final_image, result, classification_model))
                # Predecir objetos de movilidad (sillas de ruedas, andadores, etc.)
                preds_mobility = detection_model_mobility.predict(image_redimensionada, classes=[0, 1, 2, 3, 4], conf=0.3)[0]
                boxes_special_objects = preds_mobility.boxes
                
                # Si se detectan objetos especiales, contar cuántos de cada tipo hay
                if boxes_special_objects is not None:
                    class_ids = boxes_special_objects.cls.cpu().numpy()
                    special_objects = len(class_ids)
                    wheelchairs = sum(1 for x in class_ids if x == 1)  # Clase 1: Sillas de ruedas
                    walkers = sum(1 for x in class_ids if x == 4)      # Clase 4: Andadores
                    crutches = sum(1 for x in class_ids if x == 3)     # Clase 3: Muletas
                else:
                    special_objects = 0
                    wheelchairs = 0
                    walkers = 0
                    crutches = 0
                
                # Dibujar cajas alrededor de los objetos de movilidad detectados
                for box in preds_mobility.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    label = f'{box.conf[0]:.2f}'
                    cv2.rectangle(final_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Caja verde para objetos de movilidad
                    cv2.putText(final_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                log_prediction(log_file_path, request_id, timestamp, lift_id, model_version, special_objects, wheelchairs, walkers, crutches, num_people, "; ".join(gender_classification))

            else:
                print(f"No se detectaron ambos ojos o los puntos clave no están en las zonas de interés para la persona {i}. Clasificación cancelada.")

        # Guardar las predicciones en el archivo de log
    else:
        print("No se detectaron personas, ejecución cancelada.")

    # Mostrar la imagen final con predicciones y líneas dibujadas
    cv2.imshow("Resultado", final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejecución principal del script
if __name__ == '__main__':
    # Rutas de la imagen y archivo de log, además de IDs y versión de modelo
    image_path = r"C:\Users\Jm\Desktop\HYBLICON-II\Test_05_08_24\image_34.jpg"
    log_file_path = r"C:\Users\Jm\Desktop\HYBLICON-II\HYBLICON-II\HYBLICON-Tests\ProyectoGlobal\logs\log_predictions.csv"
    lift_id = "LIFT_001"
    model_version = "v1.0"
    
    # Llamada a la función de predicción
    predict_image(image_path, log_file_path, lift_id, model_version)
