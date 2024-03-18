import cv2
import numpy as np

def detect_objects(image_path):
    # Charger le modèle YOLOv3 pré-entraîné
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    
    # Obtenir les noms des couches de sortie
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # Charger des classes
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f]

    # Charger l'image
    img = cv2.imread(image_path)
    height, width, channels = img.shape

    # Prétraitement des images
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Informations de détection
    class_ids = []
    confidences = []
    boxes = []

    # Analyser les sorties du réseau
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Coordonnées du cadre de délimitation
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Coordonnées du cadre de délimitation (coin supérieur gauche)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Supprimer les détections redondantes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Afficher les résultats
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Afficher l'image résultante
    cv2.imshow("Object Detection", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


detect_objects('C:/Users/user/OneDrive/Bureau/Mes projets/Python/food_wine.jpg')
