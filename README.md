# TB-DETR
## Title
Compact Structural Feature Enhancement for Unsupervised Anomaly Detection in Chest Radiographs
## Data
The dataset used in the paper can be found here：

http://air.ug/microscopy/


## Dependencies

conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia

## Train
``` 
from ultralytics.models import RTDETR

if __name__ == '__main__':
    model=RTDETR(model='usr/TB-DETR/ultralytics/cfg/models/your.yaml')
    model.train(pretrained=False,data='usr/TB-DETR/ultralytics/cfg/datasets/your.yaml',epochs=300,device=0,batch=4,imgsz=640,workers=8,optimizer='SGD')
```
## Test
```
from ultralytics.models import RTDETR

if __name__ == '__main__':
    model=RTDETR(model='usr/TB-DETR/ultralytics/runs/detect/train/weights/best.pt')
    model.val(data='usr/TB-DETR/ultralytics/cfg/datasets/your.yaml',device=0,batch=4,imgsz=640,workers=8)
```

