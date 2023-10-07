# Obstacle Detection
## Installation
Docker environment (recommended)  
Configuration Reference : **requirements.txt**

## Testing
<details open>
<summary>bash</summary>
python test.py --data data/coco.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights yolov7_tiny.pt --name yolov7_640_val
</details>
Verify that the weights file works

## Training
### Data preparation
<details open>
<summary>bash</summary>
bash scripts/get_coco.sh
</details>
Download MS COCO dataset images (train, val, test) and labels. If you have previously used a different version of YOLO, we strongly recommend that you delete train2017.cache and val2017.cache files, and redownload labels

### Single GPU training
train p5 models
<details open>
<summary>bash</summary>
python train.py --workers 8 --device 0 --batch-size 32 --data data/coco.yaml --img 640 640 --cfg cfg/training/yolov7.yaml --weights '' --name yolov7 --hyp data/hyp.scratch.p5.yaml
</details>

## Transfer learning
### Single GPU finetuning for custom dataset
finetune p5 models
<details open>
<summary>bash</summary>
python train.py --workers 8 --device 0 --batch-size 32 --data data/custom.yaml --img 640 640 --cfg cfg/training/yolov7-custom.yaml --weights 'yolov7_training.pt' --name yolov7-custom --hyp data/hyp.scratch.custom.yaml
</details>

## Inference
On video:
<details open>
<summary>bash</summary>
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source yourvideo.mp4
</details>

On image:
<details open>
<summary>bash</summary>
python detect.py --weights yolov7.pt --conf 0.25 --img-size 640 --source inference/images/horses.jpg
</details>

## Export
Pytorch to ONNX with NMS (and inference)
<details open>
<summary>bash</summary>
python export.py --weights yolov7-tiny.pt --grid --end2end --simplify \
        --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 640
</details>

## Predict
Calling the camera for real-time detection
<details open>
<summary>bash</summary>
python main.py
</details>