import os
import glob
import numpy as np
import cv2 as cv
import json

# --- ユーザー指定 --- 
# ここをそのまま使ってOK（例: '/image/*.jpg'）
file_path = 'G:/dcon/ir-camera/calib_img/*.JPG'       # file の場所を確認して

# オプション: 歪み補正のサンプル
img_path = 'G:/dcon/ir-camera/calib_img/calib_img_15.JPG'

# 保存場所
save_path = "G:/dcon/ir-camera/ir_camera_calib.npz"


# チェスボード内部コーナー数（行, 列）
cbrow = 6
cbcol = 12

# サブピクセル収束条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# オブジェクトポイント（Z=0）。必要なら square_size を掛けて実世界単位にする
objp = np.zeros((cbrow * cbcol, 3), np.float32)
objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)
square_size = 22.0  # mm など
objp[:, :2] *= square_size

objpoints = []
imgpoints = []

# ファイル一覧取得（file_path をそのまま使う）
images = sorted(glob.glob(file_path))
if len(images) == 0:
    print(f"No images found for pattern: {file_path}")
else:
    for fname in images:
        img = cv.imread(fname)
        if img is None:
            print("読み込めませんでした:", fname)
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # findChessboardCorners の patternSize は (cols, rows) = (cbcol, cbrow)
        ret, corners = cv.findChessboardCorners(gray, (cbcol, cbrow), None)

        if ret:
            objpoints.append(objp.copy())
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            cv.drawChessboardCorners(img, (cbcol, cbrow), corners2, ret)
            cv.imshow('chessboard_corners', img)
            # 500ms 表示。Esc キーで早期終了
            if cv.waitKey(50) & 0xFF == 27:
                print("ユーザにより中断")
                break
        else:
            print("見つけられませんでした:", fname)

    cv.destroyAllWindows()

print("チェック完了")

# チェック: 十分な検出が行われたか
if len(objpoints) == 0 or len(imgpoints) == 0:
    raise RuntimeError("objpoints または imgpoints が空です。角点検出が成功しているか確認してください。")

# 画像サイズ: 最後に使ったグレースケール画像から取得するのが確実
img_shape = gray.shape[::-1]  # (width, height)

# キャリブレーション実行
retval, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_shape, None, None)

print("再投影誤差:", retval)
print("カメラ行列:\n", mtx)
print("歪み係数:\n", dist)

# 回転・並進ベクトルを見やすく表示する（DataFrame があれば便利）
import pandas as pd
rvecs_list = [vec.flatten() for vec in rvecs]
tvecs_list = [vec.flatten() for vec in tvecs]
df_r = pd.DataFrame(rvecs_list, columns=['X','Y','Z'])
df_t = pd.DataFrame(tvecs_list, columns=['X','Y','Z'])
print("回転ベクトル:\n", df_r)
print("並進ベクトル:\n", df_t)

# 保存用の辞書に変換（numpy 配列は list に変換）
data = {
    "camera_matrix": mtx.tolist(),
    "dist_coeff": dist.tolist(),
    "rvecs": [rvec.flatten().tolist() for rvec in rvecs],
    "tvecs": [tvec.flatten().tolist() for tvec in tvecs],
    "img_shape": list(img_shape)
}

# 保存パス
json_save_path = "G:/dcon/ir-camera/focal_mediapipe.json"

with open(json_save_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"JSON ファイルとして保存しました: {json_save_path}")

# 保存
#np.savez(save_path, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, img_shape=img_shape)

calib = np.load(save_path)
mtx = calib["mtx"]
dist = calib["dist"]
rvecs = calib["rvecs"]
tvecs = calib["tvecs"]
img_shape = tuple(calib["img_shape"])


# 例: rvecs, tvecs, imgpoints, objpoints は calibrateCamera 後に得られている
mean_errors = []
for i in range(len(objpoints)):
    imgpt2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    imgpt2 = imgpt2.reshape(-1,2)
    obs = imgpoints[i].reshape(-1,2)
    err = np.linalg.norm(obs - imgpt2, axis=1).mean()
    mean_errors.append(err)
# 出力
for i,e in enumerate(mean_errors):
    print(f"image {i}: mean reproj err = {e:.3f} px")
print("overall mean:", np.mean(mean_errors))




# 画像を読み込む（重要）
img = cv.imread(img_path)
if img is None:
    raise FileNotFoundError(f"画像を読み込めませんでした: {img_path}")

# リサイズ設定
target_width = 1400
h, w = img.shape[:2]
scale = target_width / float(w)
new_w = int(w * scale)
new_h = int(h * scale)
img = cv.resize(img, (w, h), interpolation=cv.INTER_AREA)  # 縮小向けに INTER_AREA
for i in range(len(rvecs)):
    R, _ = cv.Rodrigues(rvecs[i])
    print(f"Image {i}:\nR=\n{R}\nt=\n{tvecs[i].reshape(-1)}\n")


newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, img_shape, 1, img_shape)
undistorted = cv.undistort(img, mtx, dist, None, newcameramtx)
cv.imshow('undistorted', undistorted); cv.waitKey(0); cv.destroyAllWindows()