"""外接/本机摄像头实时看 YOLO-OBB 效果。本机当前只有编号 0。."""

import cv2

from ultralytics import YOLO

WEIGHTS = r"E:\XMY\代码\ultralytics\runs\train\11n_100_deg45\weights\best.pt"
CAMERA_ID = 0  # 这台电脑探测到只有 0；没有 1

model = YOLO(WEIGHTS)
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"打不开摄像头 {CAMERA_ID}。请关掉占用摄像头的软件后重试。")

print("摄像头已打开，按 q 退出")
while True:
    ok, frame = cap.read()
    if not ok:
        print("读帧失败")
        break
    result = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)[0]
    vis = result.plot()
    cv2.imshow("YOLO-OBB", vis)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
