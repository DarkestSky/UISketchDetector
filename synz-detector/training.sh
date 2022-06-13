IMAGES_PATH="./data/SynZ_Download/images"
ANNOTATIONS_PATH="./data/SynZ_Download/annotations"


NUM_GPUS=1
CONFIG_FILE="configs/SynZ-RetinaNet.yaml"
MODEL_PATH="COCO-Detection/retinanet_R_50_FPN_1x.yaml"
OUTPUT_PATH="models/synz_training/train"

CUDA_VISIBLE_DEVICES=1 python ./detectron.py \
  --num-gpus=$NUM_GPUS  \
  --config-file=$CONFIG_FILE \
  --model-path=$MODEL_PATH \
  --images-path=$IMAGES_PATH \
  --annotations-path=$ANNOTATIONS_PATH \
  --output-path=$OUTPUT_PATH \
  --resume \
  DATASETS.TEST "('synz_test', )" \
  MODEL.WEIGHTS "./models/synz_checkpoint/model_final.pth"
