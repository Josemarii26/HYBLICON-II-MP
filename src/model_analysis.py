import cv2

def analyze_bboxes(image, preds, model):
    """
    Analiza las bounding boxes detectadas en una imagen, clasificando el género y edad
    de las personas u objetos dentro de esas cajas usando un modelo de clasificación.

    Devuelve una lista de etiquetas clasificadas (género y edad) para cada caja en la imagen.
    """
    gender_classification = []  # Lista para almacenar las clasificaciones de género o categoría

    # Iterar sobre las cajas delimitadoras (bounding boxes) de las predicciones
    for i, bbox in enumerate(preds.boxes):
        # Extraer las coordenadas (x1, y1, x2, y2) de la caja delimitadora y convertirlas a enteros
        x1, y1, x2, y2 = map(int, bbox.xyxy[0].cpu().numpy())
        
        # Recortar la porción de la imagen correspondiente a la caja delimitadora
        cropped_img = image[y1:y2, x1:x2]
        
        # Hacer una predicción sobre la imagen recortada usando el modelo de clasificación
        results = model.predict(cropped_img)
        
        # Obtener el índice de la clase predicha (top1) y su confianza
        predicted_class_index = results[0].probs.top1
        predicted_class_conf = results[0].probs.top1conf.item()  # Convertir a valor flotante
        
        # Obtener los nombres de las clases del modelo
        class_names = results[0].names
        
        # Obtener el nombre de la clase predicha usando el índice
        predicted_class_name = class_names[predicted_class_index]

        # Añadir la clasificación de género o categoría a la lista
        gender_classification.append(predicted_class_name)
        
        # Preparar etiquetas para mostrar en la imagen: clase y confianza
        label_class = f'{predicted_class_name}'  # Texto de la clase
        label_conf = f'{predicted_class_conf:.2%}'  # Texto de la confianza (en porcentaje)
        
        # Parámetros de visualización: escala de fuente y grosor
        font_scale = 0.3
        font_thickness = 1
        
        # Obtener tamaño del texto para ajustar la posición de las etiquetas en la imagen
        (text_width_class, text_height_class), baseline_class = cv2.getTextSize(label_class, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        (text_width_conf, text_height_conf), baseline_conf = cv2.getTextSize(label_conf, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        
        # Espacio adicional entre las dos líneas de texto
        additional_space = 5

        # Posiciones donde se colocarán las etiquetas sobre la imagen
        text_x = x1 + 5  # Coordenada X para el texto
        text_y_class = y2 - 5 - text_height_conf - additional_space  # Coordenada Y para la clase
        text_y_conf = y2 - 5  # Coordenada Y para la confianza
        
        # Calcular la altura total necesaria para ambas etiquetas (clase + confianza)
        total_text_height = text_height_class + text_height_conf + baseline_class + baseline_conf + additional_space
        
        # Dibujar un rectángulo blanco detrás del texto para mejorar la legibilidad
        cv2.rectangle(image, (x1, y2 - total_text_height - 5), (x1 + max(text_width_class, text_width_conf) + 10, y2), (255, 255, 255), -1)
        
        # Colocar el texto de la clase sobre la imagen
        cv2.putText(image, label_class, (text_x, text_y_class), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)
        
        # Colocar el texto de la confianza sobre la imagen
        cv2.putText(image, label_conf, (text_x, text_y_conf), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), font_thickness)
    
    return gender_classification  
