from retinaface import RetinaFace
import cv2
import face_recognition
import mediapipe as mp

image_path = 'image/C.jpg'
all_faces = []
img = cv2.imread(image_path)
img2 = cv2.imread("segmented_image2.jpg")

facesR = RetinaFace.detect_faces(image_path)
for key in facesR.keys():
    identity = facesR[key]
    facial_area = identity["facial_area"]
    all_faces.append((facial_area[1], facial_area[2], facial_area[3], facial_area[0]))



face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
facesH = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
for (x, y, w, h) in facesH:
    all_faces.append((y, x + w, y + h, x))


image_for_recognition = face_recognition.load_image_file(image_path)
face_locations = face_recognition.face_locations(image_for_recognition)
for (top, right, bottom, left) in face_locations:
    all_faces.append((top, right, bottom, left))


mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
facesM = face_detection.process(rgb_img)
if facesM.detections:
    for detection in facesM.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = img.shape
        (x, y, w, h) = (int(bboxC.xmin * iw), int(bboxC.ymin * ih), 
                        int(bboxC.width * iw), int(bboxC.height * ih))
        all_faces.append((y, x + w, y + h, x))



for (top, right, bottom, left) in all_faces:
    cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
output_path = 'output1.jpg'
cv2.imwrite(output_path, img)
