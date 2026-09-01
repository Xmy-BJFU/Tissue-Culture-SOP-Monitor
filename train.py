import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # YOLO26n 旋转框（OBB）。若要用官方预训练权重，改成 YOLO('yolo26n-obb.pt')
    model = YOLO(r"E:\XMY\代码\ultralytics\ultralytics\cfg\models\11\yolo11-obb.yaml")
    # model.load('yolo26n-obb.pt')
    model.train(
        data=r"E:\XMY\代码\ultralytics\data.yaml",
        imgsz=640,
        epochs=100,
        batch=8,
        workers=0,
        device=0,
        optimizer="SGD",
        close_mosaic=10,
        resume=False,
        project=r"E:\XMY\代码\ultralytics\runs\train",
        name="11_100_deg45",
        single_cls=False,
        cache=False,
        degrees=30,  # OBB 旋转增强，让模型不怕物体转动
    )
