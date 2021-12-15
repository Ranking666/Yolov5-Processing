# Yolov5-Processing
## accomplished
- 2021.12.15
  - change backbone to Ghostnet
  - Finish EagleEye pruning YOLOv5 series

## Requirements
```
pip install -r requirements.txt
```
## Usage
### Ghostnet
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
|yolov5s_eagleeye|30.0|15.5|8.6 |8.0|
## TO DO
+ [ ] backbone: ShuffleNetV2
+ [ ] backbone: EfficientNet
+ [ ] backbone: SwinTrans
+ [ ] Prune: Other Algorithms
+ [ ] Quantization
+ [ ] Knowledge Distillation 



