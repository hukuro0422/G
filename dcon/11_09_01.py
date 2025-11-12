import cv2
import mediapipe as mp
import math
import json
import numpy as np

# ---- 設定（必要に応じて変えてね）----
KNOWN_DISTANCE = 0.40     # カメラから顔までの距離（m）
REAL_IPD = 0.063          # 実際の瞳孔間距離（m） 平均63mm
SAVE_FILE = "G:/dcon/ir-camera/focal_mediapipe.json"
# ---------------------------------------


with open(SAVE_FILE, "r") as f:
    data = json.load(f)
mtx = np.array(data["camera_matrix"])
dist = np.array(data["dist_coeff"])
rvecs = [np.array(rv).reshape(3,1) for rv in data["rvecs"]]
tvecs = [np.array(tv).reshape(3,1) for tv in data["tvecs"]]
img_shape = tuple(data["img_shape"])
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)

print("正面を向いて、距離 {:.2f}m に立ってください。".format(KNOWN_DISTANCE))
input("準備できたら Enter を押してください...")

ret, frame = cap.read()
frame_u = cv2.undistort(frame, mtx, dist)
h, w, _ = frame_u.shape

res = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))



if not res.multi_face_landmarks:
    print("顔が検出できませんでした。もう一度実行してね。")
else:
    lm = res.multi_face_landmarks[0].landmark
    xL, yL = lm[33].x * w, lm[33].y * h
    xR, yR = lm[263].x * w, lm[263].y * h
    pixel_ipd = math.dist((xL, yL), (xR, yR))

    focal_length = (pixel_ipd * KNOWN_DISTANCE) / REAL_IPD
    print("推定した焦点距離:", focal_length)

    data = {
        "camera_matrix": mtx.tolist(),
        "dist_coeff": dist.tolist(),
        "focal": focal_length,
        "real_ipd": REAL_IPD,
        "rvecs": [rvec.flatten().tolist() for rvec in rvecs],
        "tvecs": [tvec.flatten().tolist() for tvec in tvecs],
        "img_shape": list(img_shape)
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f)

    print("\n保存しました →", SAVE_FILE)

cap.release()
cv2.destroyAllWindows()
