IMAGES_PATH="./data/SynZ_Download/images"
ANNOTATIONS_PATH="./data/SynZ_Download/annotations"

CUDA_VISIBLE_DEVICES=3

NUM_GPUS=1
CONFIG_FILE="configs/SynZ-RetinaNet.yaml"
MODEL_PATH="COCO-Detection/retinanet_R_50_FPN_1x.yaml"
OUTPUT_PATH="models/synz_training/evaluation_on_val"

python ./detectron.py \
  --num-gpus=$NUM_GPUS  \
  --config-file=$CONFIG_FILE \
  --model-path=$MODEL_PATH \
  --images-path=$IMAGES_PATH \
  --annotations-path=$ANNOTATIONS_PATH \
  --output-path=$OUTPUT_PATH \
  --resume \
  --eval-only \
  MODEL.WEIGHTS "models/synz_checkpoint/model_final.pth" \
  DATASETS.TEST "('synz_val', )" \
