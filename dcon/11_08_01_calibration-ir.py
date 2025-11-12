import cv2
import os


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("カメラが見つかりませんでした…")
    exit()

count = 0  # 保存ファイルの番号

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1)

    if key == 32:  # スペースキー
        filename = f"G:/dcon/ir-camera/calib_img/calib_img_{count}.jpg"
        cv2.imwrite(filename, frame)
        print(f"保存しました: {filename}")
        count += 1
    

    elif key == 27:  # ESC で終了
        break

cap.release()
cv2.destroyAllWindows()
