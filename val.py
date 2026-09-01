import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 换成你训练得到的权重，一般在 runs/train/exp/weights/best.pt
    model = YOLO(model='runs/train/exp/weights/best.pt')
    metrics = model.val(data='data.yaml',
                        split='val',
                        imgsz=640,
                        batch=4,
                        workers=0,
                        device='',
                        conf=0.25,
                        iou=0.7,
                        plots=False,
                        project='runs/val',
                        name='exp',
                        )
    print(f'mAP50-95: {metrics.box.map:.4f}')
    print(f'mAP50: {metrics.box.map50:.4f}')
    print(f'mAP75: {metrics.box.map75:.4f}')
    print(f'各类别 mAP50-95: {metrics.box.maps}')
