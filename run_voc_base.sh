gpu_id=$4
echo "gpu_id: ${gpu_id}"
for base in 1 2 3
do   
    config_file="configs/voc/ecea_det_r101_base${base}.yaml"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --config-file ${config_file} \
        --output_dir "voc/base${base}" \
        --num-gpus  4 \
done

