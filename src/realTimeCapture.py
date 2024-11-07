import cv2
import threading
import queue
import time
from ultralytics import YOLO
from datetime import datetime
from utils import generate_random_id, log_prediction
from image_processing import evaluar_calidad, redimensionar_imagen
from model_analysis import analyze_bboxes

# Definir una cola para los frames
frame_queue = queue.Queue(maxsize=10)  # Limita la cola a 10 frames para evitar retrasos excesivos

# Event para indicar cuándo detener los hilos
stop_event = threading.Event()

def frame_capture(cap, frame_queue, stop_event):
    """
    Hilo encargado de capturar frames de la cámara y colocarlos en la cola.
    """
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("Error: No se pudo capturar la imagen desde la cámara.")
            stop_event.set()
            break
        try:
            # Si la cola está llena, descartar el frame más antiguo
            if frame_queue.full():
                discarded_frame = frame_queue.get_nowait()
                frame_queue.put(frame)
            else:
                frame_queue.put(frame)
        except queue.Full:
            pass  # En caso de que otro hilo lo haya llenado
    print("Frame capture thread terminado.")

def frame_processing(log_file_path, lift_id, model_version, frame_queue, stop_event):
    """
    Hilo encargado de procesar los frames de la cola.
    """
    # Cargar los modelos necesarios
    detection_model_person = YOLO("./models/bestY11-pose-n.pt")  # Modelo de detección de personas
    classification_model = YOLO('./models/bestY11-UTFK-v2.pt')  # Modelo de clasificación de género
    detection_model_mobility = YOLO('./models/bestY11-v1-n.pt')  # Modelo de detección de objetos de movilidad

    while not stop_event.is_set():
        try:
            # Esperar hasta que haya un frame disponible o hasta que se detenga
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue  # No hay frames para procesar en este momento

        start_time = time.time()

        # Redimensionar la imagen y evaluar su calidad
        frame_redimensionado = redimensionar_imagen(frame)
        

        # Generar un ID único y obtener la marca de tiempo actual
        request_id = generate_random_id()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Realizar predicciones de personas en la imagen usando el modelo de detección
        preds_person = detection_model_person.predict(frame_redimensionado, conf=0.4)[0]
        num_people = len(preds_person.boxes)  # Número de personas detectadas
        final_frame = preds_person.plot()  # Imagen con las predicciones dibujadas

        gender_classification = []  # Lista para almacenar las clasificaciones de género

        # Verificar si se detectaron personas
        if num_people > 0:
            # Obtener dimensiones de la imagen
            height, width = frame_redimensionado.shape[:2]

            # Definir límites para las zonas de interés en la imagen (1/3 y 2/3 de la altura)
            lower_limit_1_3 = height / 3
            upper_limit_1_3 = (2 * height) / 3
            lower_limit_2_3 = upper_limit_1_3
            upper_limit_2_3 = height

            print("Se detectan personas en la imagen.")

            # Dibujar líneas horizontales que dividen la imagen en tercios
            cv2.line(final_frame, (0, int(lower_limit_1_3)), (width, int(lower_limit_1_3)), (0, 0, 255), 2)  # Línea a 1/3 de altura (Rojo)
            cv2.line(final_frame, (0, int(upper_limit_1_3)), (width, int(upper_limit_1_3)), (0, 0, 255), 2)  # Línea a 2/3 de altura (Rojo)

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
                    gender_classification.extend(analyze_bboxes(final_frame, result, classification_model))
                    # Predecir objetos de movilidad (sillas de ruedas, andadores, etc.)
                    preds_mobility = detection_model_mobility.predict(frame_redimensionado, classes=[0, 1, 2, 3, 4], conf=0.6)[0]
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
                        cv2.rectangle(final_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Caja verde para objetos de movilidad
                        cv2.putText(final_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    log_prediction(
                        log_file_path,
                        request_id,
                        timestamp,
                        lift_id,
                        model_version,
                        special_objects,
                        wheelchairs,
                        walkers,
                        crutches,
                        num_people,
                        "; ".join(gender_classification)
                    )

                else:
                    print(f"No se detectaron ambos ojos o los puntos clave no están en las zonas de interés para la persona {i}. Clasificación cancelada.")

        else:
            print("No se detectaron personas, ejecución cancelada.")

        # Mostrar el cuadro procesado con predicciones y líneas dibujadas
        cv2.imshow("Resultado", final_frame)

        # Salir del bucle si se presiona la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()

        processing_time = time.time() - start_time
        print(f"Tiempo de procesamiento de este frame: {processing_time:.2f} segundos")

    print("Frame processing thread terminado.")

def predict_from_camera_async(log_file_path, lift_id, model_version):
    """
    Función principal que configura los hilos de captura y procesamiento de frames.
    """
    raspberry_ip_1 = "192.168.0.101"
    stream_url_1 = f"tcp://{raspberry_ip_1}:8080"
    #stream_url_1= f'udp://{raspberry_ip_1}:8080'

    cap = cv2.VideoCapture(stream_url_1)
    #cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return

    # Obtener las dimensiones del video (ancho y alto)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10  # Definir el número de fotogramas por segundo (ajústalo según tus necesidades)

    # Opcional: Definir el codec y crear los objetos VideoWriter para el original y procesado
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec para el video (puedes usar 'XVID' o 'MJPG')

    # Video original
    #original_video_writer = cv2.VideoWriter(f'original_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi', fourcc, fps, (width, height))

    # Video con resultados procesados
    #processed_video_writer = cv2.VideoWriter(f'processed_{datetime.now().strftime("%Y%m%d_%H%M%S")}.avi', fourcc, fps, (width, height))

    # Crear y iniciar los hilos
    capture_thread = threading.Thread(target=frame_capture, args=(cap, frame_queue, stop_event))
    processing_thread = threading.Thread(target=frame_processing, args=(log_file_path, lift_id, model_version, frame_queue, stop_event))

    capture_thread.start()
    processing_thread.start()

    try:
        while not stop_event.is_set():
            time.sleep(1)  # Mantener el hilo principal activo
    except KeyboardInterrupt:
        print("Interrupción manual recibida. Deteniendo hilos...")
        stop_event.set()

    # Esperar a que los hilos terminen
    capture_thread.join()
    processing_thread.join()

    # Liberar los recursos de la cámara y cerrar ventanas
    cap.release()
    #original_video_writer.release()  # Liberar el VideoWriter del video original
    #processed_video_writer.release()  # Liberar el VideoWriter del video procesado
    cv2.destroyAllWindows()

# Ejecución principal del script
if __name__ == '__main__':
    log_file_path = r"C:\Users\Jm\Desktop\ProyectoGlobal\logs\log_predictions.csv"
    lift_id = "LIFT_001"
    model_version = "v1.0"

    predict_from_camera_async(log_file_path, lift_id, model_version)
