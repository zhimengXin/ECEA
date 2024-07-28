gpu_id=$4
echo "gpu_id: ${gpu_id}"   
config_file="configs/coco/ecea_det_r101_base.yaml"
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config-file ${config_file} \
    --output_dir "coco/base" \
    --num-gpus  4 \
