while getopts "m:c:t:s:" opt;do
  case $opt in
  m)
    model=$OPTARG
    ;;
  t)
    train=${OPTARG}
    ;;
  esac
done
echo "Strategy: [${model}]"
echo "train=[${train}]"
# experiments_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/experiments/Graduate_project/two_stage2"
experiments_path="/public_bme/data/v-xiongxs/MICCAI2024/ASOCA"
data_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/data/ASOCA_New"
configs_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/configs/MICCAI2024"

############################################################# coarse segmentation ################################################################
### 1. train
if [ ${train} == "1" ]; then
python3 main_v4.py \
--mode=train \
--configs=${configs_path}/${model}.yaml \
--experiments_path=${experiments_path}/${model} \
--persist_path=${experiments_path}/persist_cache \
--img_path=${data_path}/imagesTr \
--label_path=${data_path}/labelsTr \
--val_set=${data_path}/nnUnet_val_0.txt \
--train_set=${data_path}/nnUnet_train_0.txt \
--workers=4 || { echo 'Error during training.'; exit 1; }
fi
#### 2. inference
python3 main_v4.py \
--mode=infer \
--configs=${configs_path}/${model}.yaml \
--experiments_path=${experiments_path}/${model} \
--img_path=${data_path}/imagesTs \
--output_path=${experiments_path}/${model}/test_infer \
--pretrain_weight_path=${experiments_path}/${model}/checkpoint/best_metric_model.pth \
--workers=8 || { echo 'Error during inference.'; exit 1; }
#### 3. caculate metrics
## main test, GT is available in main phase
python3 caculate_metric.py \
--seg_path=${experiments_path}/${model}/test_infer \
--label_path=${data_path}/labelsTs || { echo 'Error during caculating metrics.'; exit 1; }

