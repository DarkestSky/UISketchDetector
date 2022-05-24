CONFIG_FILE="./cfgs/refcoco/base_detected_regions_4x16G.yaml"
CHECKPOINT_FILE="./refcoco/model/test-train/output/vl-bert/synz/base_detected_regions_4x16G/train_train/vl-bert_base_res101_refcoco-best.model"
SAVE_PATH="./refcoco/test_result_pth"

python refcoco/test.py \
    --split test \
    --cfg $CONFIG_FILE \
    --ckpt $CHECKPOINT_FILE \
    --gpus 2 \
    --result-path $SAVE_PATH --result-name first