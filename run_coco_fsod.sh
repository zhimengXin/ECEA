gpu_id=$4
echo "gpu_id: ${gpu_id}"
for shot in 1 2 3 5 10 30
do   
    config_file="configs/coco/ecea_fsod_r101_novel_${shot}shot_seedx.yaml"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config-file ${config_file} \
        --output_dir "coco/${shot}shot_fsod" \
        --num-gpus  4 \
        --opts MODEL.WEIGHTS "coco/model_reset_surgery.pth" \
done
