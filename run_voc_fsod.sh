gpu_id=$4
echo "gpu_id: ${gpu_id}"
for shot in 1 2 3 5 10
do 
    config_file="configs/voc/ecea_gfsod_r101_novel1_${shot}shot_seed1.yaml"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config-file ${config_file} \
        --output_dir "base1/${shot}shot_gfsod" \
        --num-gpus  4 \
        --opts MODEL.WEIGHTS "voc/model_reset_surgery.pth" \
done

