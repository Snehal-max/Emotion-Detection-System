import cv2
import numpy as np
from tensorflow.keras.models import load_model

model        = load_model("model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def predict_emotion(face_gray):
    face       = cv2.resize(face_gray, (48, 48))
    face       = face / 255.0
    face       = np.reshape(face, (1, 48, 48, 1))
    prediction = model.predict(face, verbose=0)
    emotion    = labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    return emotion, confidence

cap = cv2.VideoCapture(0)

print("Webcam started — Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor = 1.3,
        minNeighbors= 5
    )

    for (x, y, w, h) in faces:
        face_crop           = gray[y:y+h, x:x+w]
        emotion, confidence = predict_emotion(face_crop)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{emotion} {confidence:.1f}%",
            (x, y-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam closed")