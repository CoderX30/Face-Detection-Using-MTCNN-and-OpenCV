from mtcnn import MTCNN
import cv2

for i in range(1,11):

    image_path = f'Image {i}.jpeg'
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detections = detector.detect_faces(img)
    detections


    import matplotlib.pyplot as plt

    img_with_dets = img.copy()
    min_conf = 0.71
    for det in detections:
        if det['confidence'] >= min_conf:
            x, y, width, height = det['box']
            keypoints = det['keypoints']
            cv2.rectangle(img_with_dets, (x,y), (x+width,y+height), (0,155,255), 2)

    plt.figure(figsize = (10,10))
    plt.imshow(img_with_dets)
    plt.axis('off')