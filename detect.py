import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # 换成你训练得到的权重，一般在 runs/train/exp/weights/best.pt
    model = YOLO(model=r'E:\XMY\代码\ultralytics\runs\train\100\weights\best.pt')
    model.predict(source=r'E:\XMY\data\images\val',  # 单张图片、文件夹或视频路径
                  imgsz=640,
                  conf=0.25,
                  iou=0.7,
                  device='',
                  project=r'E:\XMY\代码\ultralytics\runs\detect',
                  name='exp',
                  save=True,        # 保存画好检测框的图片
                  save_txt=True,    # 保存检测框坐标到 txt
                  save_conf=True,   # txt 中同时保存置信度
                  show=False,
                  )
