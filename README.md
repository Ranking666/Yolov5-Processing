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
#### FPGM / SFP
cd yolov5_fpgm_slimming_sfp
1. soft mask
```
python train.py --data data/VisDrone.yaml --imgsz 640 --weights yolov5s.pt --cfg models/yolov5s_pruning.
yaml --device 0,1,2,3 --sfp/fpgm --sfp_ratio/fpgm_ratio 0.5 --path models/yolov5s_pruned.yaml
```
2. Fine_tuning
```
python train.py --data data/VisDrone.yaml --imgsz 640 --weights pruned_weights.pt --cfg models/yolov5s_fpgm/sfp_pruned.yaml --device 0,1,2,3
```
#### network_slimming
cd yolov5_fpgm_slimming_sfp
1. BatchNorm Layer \gamma 
```
python train.py --data data/VisDrone.yaml --imgsz 640 --weights yolov5s.pt --cfg models/yolov5s_pruning.yaml --device 0,1,2,3 -sr 
```
2. BatchNorm Layer pruning
```
python prune_slimming.py --weights the_first_step_trained_model --data data/VisDrone.yaml --device 0
```
3. Fine_tuning
```
python train.py --data data/VisDrone.yaml --imgsz 640 --weights the_second_step_get_model --cfg models/yolov5s_slimm_pruned.yaml --device 0,1,2,3
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



