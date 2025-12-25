import warnings

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('/root/autodl-tmp/yolo/MFA-YOLO/ultralytics/cfg/models/v8/MFA-YOLO.yaml')
    model.train(data='/root/autodl-tmp/Datasets/NWPU VHR-10/nwpu.yaml',
                cache=False,
                imgsz=1024,
                epochs=200,
                batch=16,
                close_mosaic=0,
                workers=8,
                optimizer='SGD',
                project='runs/train',
                name='exp',
                )
