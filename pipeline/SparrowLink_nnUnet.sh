#### nnunetv2.yaml only support torch >= 2.0.0

export nnUNet_raw="/public/home/v-xiongxx/Graduate_project/nnUnetv2/nnUNet/data_and_result/nnunetv2_raw"
export nnUNet_preprocessed="/public/home/v-xiongxx/Graduate_project/nnUnetv2/nnUNet/data_and_result/nnunetv2_preprocessed"
export nnUNet_results="/public/home/v-xiongxx/Graduate_project/nnUnetv2/nnUNet/data_and_result/nnunetv2_results"

experiments_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/experiments/Graduate_project/two_stage2"
data_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/data/multi_phase_select/final_version_crop_train"
configs_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/configs/two_stage2"

task_id=800
model="nnunetv2"
### data conversion
#python3 /public/home/v-xiongxx/Graduate_project/nnUnetv2/nnUNet/nnunetv2.yaml/dataset_conversion/Dataset800_CCTA1.py \
#--imagestr_path=${data_path}/train/main/img_crop \
#--labelstr_path=${data_path}/train/main/label_crop_clip \
#--imagests_path=${data_path}/test/main/img_crop \
#--labelsts_path=${data_path}/test/main/label_crop_clip \
#--imagestr_auxiliary_path=${data_path}/train/auxiliary/img_crop \
#--imagests_auxiliary_path=${data_path}/test/auxiliary/img_crop \
#--d=${task_id}

#nnUNetv2_plan_and_preprocess -d ${task_id} --verify_dataset_integrity
#nnUNetv2_train ${task_id} 3d_fullres 0 --npz
task_id=800
task_name=Dataset800_CCTA1
## test
#declare -A data_fold
#data_fold=(["test_main"]="imagesTs" ["train_main"]="imagesTr" ["test_auxiliary"]="imagesTs_auxiliary" ["train_auxiliary"]="imagesTr_auxiliary")
#for mode in "test" ;
#do
#  for phase in "main" ;
#  do
#    nnUNetv2_predict \
#    -i ${nnUNet_raw}/${task_name}/${data_fold[${mode}_${phase}]} \
#    -o ${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_origin_nnunet \
#    -d ${task_id} \
#    -c 3d_fullres \
#    -f 0 \
#    --verbose \
#    --save_probabilities || { echo 'Error during nnUnet inference.'; exit 1; }
#  done
#done

for mode in "test" ;
do
  for phase in "main" ;
  do
    python3 main.py \
    --mode=infer \
    --configs=${configs_path}/first_stage/${model}.yaml \
    --experiments_path=${experiments_path}/first_stage/${model} \
    --img_path=${data_path}/${mode}/${phase}/img_crop \
    --output_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_try_invertd_bilinear || { echo 'Error during inference.'; exit 1; }
  done
done

#python3 caculate_metric.py \
#--seg_path=/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/transform/transform/test_transform/label_resample_resample_uint8/ \
#--label_path=${data_path}/test/main/label_crop_clip \


##### 3. caculate metrics
### main test, GT is available in main phase
python3 caculate_metric.py \
--seg_path=${experiments_path}/first_stage/${model}/main_test_infer_try_invertd_bilinear \
--label_path=${data_path}/test/main/label_crop_clip \

#while getopts "m:c:t:" opt;do
#  case $opt in
#  m)
#    model=$OPTARG
#    ;;
#  c)
#    CS=${OPTARG:0:1} # coarse segmentation
#    IP=${OPTARG:1:1} # information processing
#    RS=${OPTARG:2:1} # refined segmentation
#    IF=${OPTARG:3:1} # final inference
#    ;;
#  t)
#    train=${OPTARG}
#  esac
#done
#echo "SparrowLink using backbone: [${model}]"
#echo "coarse segmentation=[${CS}]"
#echo "information processing=[${IP}]"
#echo "refined segmentation=[${RS}]"
#echo "final inference=[${IF}]"
#echo "train=[${train}]"
#task_name="Task800_two_stage"
#task_num=800
###################################################### coarse segmentation #####################################################
### 1. train
#if [ ${CS} == "1" ]; then
#if [ ${train} == "1" ]; then
#nnUNet_plan_and_preprocess -t 800
#nnUNet_train 3d_fullres nnUNetTrainerV2 800 0
#fi
### 2. test
#img_fold=(["test_main"]="imageTs" ["train_main"]="imageTr" ["test_auxiliary"]="imageTs_auxiliary" ["train_auxiliary"]="imageTr_auxiliary")
#for mode in "train" "test" ;
#do
#  for phase in "main" "auxiliary" ;
#  data_fold=""
#  do
#    nnUNet_predict2 \
#    -i ${nnUNet_raw_data_base}/nnUNet_raw_data/${task_name}/${data_fold[${mode}_${phase}]} \
#    -o ${data_path}/first_stage/${model}/{phase}_{mode}_infer \
#    -t ${task_num} \
#    -m 3d_fullres \
#    --save_npz || { echo 'Error during nnUnet inference.'; exit 1; }
#  done
#done
#fi
##################################################### information processing block ####################################################
#if [ ${IP} == "1" ]; then
#### 1. fracture detection,
### fracture detection output is saved in the same folder as the input
#for mode in "train" "test" ;
#do
#  for phase in "main" "auxiliary" ;
#  do
#    python3 post_processing/fracture_detection.py \
#    --detection_stage=1 \
#    --save_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_detection \
#    --S_M=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer \
#    --GT=${data_path}/${mode}/${phase}/label_crop_clip \
#    --view
#  done
#done
#
### move the discontinuity label for training
#for mode in "train" "test" ;
#do
#  for phase in "main" "auxiliary" ;
#  do
#    for postfix in "_sphere" "_sphere_GT" ;
#    do
#      echo /first_stage/${model}/${phase}_${mode}_infer_detection
#      python3 post_processing/move_file.py \
#      --data_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_detection \
#      --save_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer${postfix} \
#      --data_postfix=${postfix} || { echo 'Error during discontinuity detection.'; exit 1; }
#    done
#  done
#done
#
##### 2. registration, registration output is saved in the same folder as the input, named as
#### f"{args.net}/auxiliary_{args.mode}_infer_{args.reg_algorithm}_reg_{args.time}"
#for mode in "train" "test" ;
#do
#  python3 registration/label_registration.py \
#  --reg_algorithm=SyNRA \
#  --time=1 \
#  --mode=test \
#  --save_root=${experiments_path}/first_stage/${model}/registration \
#  --main_mask_path=${experiments_path}/first_stage/${model}/main_${mode}_infer \
#  --auxiliary_mask_path=${experiments_path}/first_stage/${model}/auxiliary_${mode}_infer \
#  --main_img_path=${data_path}/${mode}/main/img_crop \
#  --auxiliary_img_path=${data_path}/${mode}/auxiliary/img_crop || { echo 'Error during registration.'; exit 1; }
#done
#fi
################################################ refined segmentation ############################################################
