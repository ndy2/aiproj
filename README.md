[Final Project] Object Detection Implementation with Pytorch
======================

# 0. Requirements
* **Python 3.6+**
* **NumPy 1.18+**
* **PyTorch 1.5+**
* **opencv (cv2)**
* **tensorboard**
* **tensorboardX**

# 1. Hyperparameters
아래의 모든 hyperparameters는 train, test시 argument로 변경 할 수 있습니다.

## 1.1 Training
### 1.1.1 Batch size (default = 10)
Batch당 Image 개수 

### 1.1.2 Momentum (default = 0.9), Decay (default = 0.9)
Training 시 사용되는 optimizer (torch.SGD)의 설정에 사용되는 hyperparameters입니다.
Momentum은 과거 학습의 경향성을 반영하여 optimizer가 더 빠르게 최적해에 도달할 수 있도록 해주고
Decay(L2 Regularization, Weight Decay) Cost function에 Ridge Regression을 적용하여 overfitting을 방지 하도록 해줍니다.
두 값 모두 default value로 일반적으로 사용되는 0.9를 사용 할 수 있습니다.

### 1.1.3 test_interval(default = 1)
validation을 통해 학습을 조정할 epoches의 간격을 설정합니다.
default는 1로 설정합니다. 즉 매 epoch마다 validation set을 적용합니다.

### 1.1.4 Epoches (default = 160)
학습 Epoch의 횟수입니다.

### 1.1.5 Loss Scales (object, noobject, class, coord - sclaes)
Yolo-V2 Loss Logic (Loss = Classification Loss + Localization(Coordinate) Loss + Confidence Loss)의 결정에 사용되는
Hyperparameters입니다. 
defualt값으로는 (object, noobject, class, coord) = (1.0, 0.5, 1.0, 5.0)을 사용하였습니다.
자세한 내용은 보고서에 서술하겠습니다.

### 1.1.6 es_min_delta, es_patience (default = disabled (0))
Eearly Stopping 관련 parameters입니다. 
es_min_delta (float)는 현재 validation loss 와 과거 validation loss의 차이에 따라 Epoch를 모두 돌지 않고 학습을 종료할 수 있게 합니다.
es_patience (int) 는 최적의 validation loss 등장후 loss가 감소하지 않고 해당 횟수만큼 validate 과정이 진행 될 시 학습을 종료시키도록 하는 변수입니다.

## 1.2 Test
## 1.2.1 conf-threshold (default = 0.35)
bounding box에 대한 confidence가 0.35이상일시 bounding 박스를 표시합니다.

## 1.2.2 nms_threshold (default = 0.35)
인접한 grid cell에서 object에 대한 confidence들이 충돌 할 경우 하나의 bounding box만 남기고 나머지를 없애는 non-maximum-suppression
에서 충돌의 기준이 되는 iou값의 threshold로 설정됩니다. 


# 2. Train Command
```
python3 train_2016310703.py
```
```
python3 train_2016310703.py --data_path data/train_val
```
: data/train_val의 데이터에 대해 training진행
data_path 아래에는 꼭 Annotations/ JPEGImages로 구분된 directory가 있어야 합니다.

# 3. Test Command
```
python3 test_2016310703.py
```
```
python3 test_2016310703.py --input data/test
```
: data/test의 데이터에 대해 test 진행 
input아래에는 꼭 test_images directory와 test_annotations directory가 나뉘어 있어야 합니다.
이때 test_annotations아래에는 voc_xml이라는 directory가 있어야 합니다...
(test과정에서 mAP를 계산할때 test data의 annotation을 yolo txt format으로 변경하는 과정이 있어서 이렇게 두었습니다...)


# 4. Directory Structure (under directory codes)
 
before training (Defualt) 
─ codes				-top dir 

  ├── data			-data dir 
  
  │  ├── test			 
  
  │  │   ├── test_annotations	-test annotations 
  
  │  │   │   └── voc_xml	-ann voc xml format 
  
  │  │   └── test_images		-test images 
  
  │  └── train_val 
  
  │       ├── Annotations		-train annotations (voc xml format)
  │       └── JPEGImages		-train images 
  ├─ train_2016310703.py 
  ├─ test_2016310703.py 
  └─ utils_2016310703.py 
  

# 5. Reference
[**Yolo-v2-pytorch**](https://github.com/uvipen/Yolo-v2-pytorch)
[**mAP](https://github.com/Cartucho/mAP)
