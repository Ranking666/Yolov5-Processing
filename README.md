---
tags: [Import-90e4]
title: Yolov5-Processing
created: '2021-12-15T03:01:35.066Z'
modified: '2021-12-27T08:52:52.792Z'
---

# Yolov5-Processing
## accomplished
- 2021.12.15
  - change backbone to Ghostnet
  - Finish EagleEye pruning YOLOv5 series
- 2021.12.27
  - change backbone to shufflenetv2
  - change backbone to efficientnetv2

## Requirements
```
pip install -r requirements.txt
```
## Usage
### different backbone
#### such as ghostnet
```
python train.py --data data/VisDrone.yaml --imgsz 640 --weights '' --cfg models/yolo_ghostnet.yaml --nosave  --device 0,1,2,3 --sync-bn
```
 You can change depth_multiple and width_multiple to choose different yolov5 verson
### prune
#### EagleEye
1. Normal Training
```
python train.py --data data/VisDrone.yaml --imgsz 640 --weights yolov5s.pt --cfg models/yolov5s_pruning.yaml --nosave  --device 0,1,2,3 --sync-bn 
```
2. Search for Optimal Pruning Network
```
python eagleeye.py --data data/VisDrone.yaml --weight the_first_step_trained_model --cfg models/yolov5_pruning.yaml --path models/yolov5s_pruned.yaml --pruned_weights pruned_weight.pt
```
3. Fine-tuning
```
python train.py --data data/VisDrone.yaml --imgsz 640 --weights pruned_weight.pt --cfg models/yolov5s_pruned.yaml  --device 0,1,2,3 --nosave --sync-bn
```
## Results
| Models | mAP@.5| mAP@.5:.95 | GFLOPS |Parameters(M)|
| ------ | ------| -----------|--------|----------   | 
| yolov5s| 35.1  |    19.4    |    15.9   | 14.4        |
|yolov5l_Ghostnet| 33.1|18.2|42.7 |49.4|
|yolov5l_efficientnetv2|23.3|11.4|35.3|42.8|
|yolov5L_shufflenetv2|29.0|15.2|38.0|40.2|
|yolov5s_eagleeye|30.0|15.5|8.6 |8.0|
## TO DO
+ [x] backbone: ShuffleNetV2
+ [x] backbone: EfficientNetV2
+ [ ] backbone: SwinTrans
+ [ ] Prune: Other Algorithms
+ [ ] Quantization
+ [ ] Knowledge Distillation 



