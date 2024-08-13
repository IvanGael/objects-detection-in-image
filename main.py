import cv2
import numpy as np

def detect_objects(image_path, output_path):
    # Load the pre-trained YOLOv3 model
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    
    # Get the names of the output layers
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Load classes
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f]
    
    # Load the image
    img = cv2.imread(image_path)
    height, width, channels = img.shape
    
    # Preprocess the image
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    # Detection information
    class_ids = []
    confidences = []
    boxes = []
    
    # Analyze network outputs
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Bounding box coordinates
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Bounding box coordinates (top-left corner)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Remove redundant detections
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw results on the image
    for i in range(len(boxes)):
        if i in indices:
            x, y, w, h = boxes[i]
            label = classes[class_ids[i]]
            confidence = confidences[i]
            color = (0, 255, 0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Save the output image
    cv2.imwrite(output_path, img)

    # cv2.imshow("Object Detection", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

input_image = 'soft_ing.jpg'
output_image = 'output_soft_ing.jpg'
detect_objects(input_image, output_image)