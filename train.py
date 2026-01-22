from ultralytics.models import RTDETR

if __name__ == '__main__':
    model=RTDETR(model='usr/ultralytics/cfg/models/your.yaml')
    model.train(pretrained=False,data='usr/ultralytics/cfg/datasets/your.yaml',epochs=300,device=0,batch=4,imgsz=640,workers=8,optimizer='SGD')