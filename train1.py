import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    # YOLO11n 旋转框（OBB），加载官方预训练权重，不要从 yaml 从零训练
    model = YOLO("yolo11n-obb.pt")
    model.train(
        data=r"E:\XMY\代码\ultralytics\data.yaml",
        imgsz=640,
        epochs=300,
        batch=16,
        workers=0,
        device=0,
        optimizer="SGD",
        close_mosaic=10,
        resume=False,
        project=r"E:\XMY\代码\ultralytics\runs\train",
        name="11n_100_deg45（9.1）",
        single_cls=False,
        cache=False,
        degrees=45,  # OBB 旋转增强，让模型不怕物体转动
    )
