CONFIG_FILE="./cfgs/refcoco/base_detected_regions_4x16G.yaml"
MODEL_DIR="./refcoco/model/fourth-train"
LOG_DIR="./refcoco/log/fourth-train"

python refcoco/train_end2end.py \
    --cfg $CONFIG_FILE \
    --model-dir $MODEL_DIR \
    --log-dir $LOG_DIR 
    
    