import cv2
import mediapipe as mp
import math
import json

LOAD_FILE = "focal_mediapipe.json"

with open(LOAD_FILE, "r") as f:
    data = json.load(f)

FOCAL = data["focal"]
REAL_IPD = data["ipd"]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        xL, yL = landmarks[33].x * w, landmarks[33].y * h
        xR, yR = landmarks[263].x * w, landmarks[263].y * h

        pixel_ipd = math.dist((xL, yL), (xR, yR))

        if pixel_ipd > 0:
            distance = (REAL_IPD * FOCAL) / pixel_ipd
            cv2.putText(frame, f"Distance: {distance:.2f} m", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        cv2.line(frame, (int(xL), int(yL)), (int(xR), int(yR)), (255,0,0), 2)

    cv2.imshow("Distance Measure (MediaPipe)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
