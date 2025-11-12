"""import os

folder_path = 'G:/dcon/calibration_image'

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
files.sort()

# 0から名前を付けていく


for i, filename in enumerate(files, start=0):  
    ext = os.path.splitext(filename)[1]
    new_name = f"calib_img_{i}{ext}"            # 付けたいファイルの名前を決める
    src = os.path.join(folder_path, filename)
    dst = os.path.join(folder_path, new_name)
    os.rename(src, dst)

print("0から始まるリネーム完了！")"""




import cv2
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

