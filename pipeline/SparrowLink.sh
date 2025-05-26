# n represent network name and c represent the use which block, for example, c=1111 means use all three blocks
# t represent training controller, for example, t=1 means training process will be executed
while getopts "m:c:t:" opt;do
  case $opt in
  m)
    model=$OPTARG
    ;;
  c)
    CS=${OPTARG:0:1} # coarse segmentation
    IP=${OPTARG:1:1} # information processing
    RS=${OPTARG:2:1} # refined segmentation
    IF=${OPTARG:3:1} # final inference
    ;;
  t)
    train=${OPTARG}
  esac
done

echo "SparrowLink using backbone: [${model}]"
echo "coarse segmentation=[${CS}]"
echo "information processing=[${IP}]"
echo "refined segmentation=[${RS}]"
echo "final inference=[${IF}]"
echo "train=[${train}]"

experiments_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/experiments/Graduate_project/two_stage2"
data_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/data/multi_phase_select/final_version_crop_train"
configs_path="/public/home/v-xiongxx/Graduate_project/Cardio_vessel_segmentaion_based_on_monai/configs/two_stage2"

############################################################# coarse segmentation ################################################################
### 1. train
if [ ${CS} == "1" ]; then
if [ ${train} == "1" ]; then
time=$(date "+%Y-%m-%d %H:%M:%S")

output_and_error_txt_path=${experiments_path}/first_stage/${model}/output_and_error_${time}.txt
python3 main.py \
--mode=train \
--configs=${configs_path}/first_stage/${model}.yaml \
--experiments_path=${experiments_path}/first_stage/${model} \
--persist_path=${experiments_path}/first_stage/persist_cache \
--img_path=${data_path}/train/main/img_crop \
--label_path=${data_path}/train/main/label_crop_clip \
--val_set=${data_path}/train/main/val.txt \
--train_set=${data_path}/train/main/train.txt | tee -a ${output_and_error_txt_path} | || { echo 'Error during training.'; exit 1; }
fi
#### 2. inference
for mode in "train" "test" ;
do
  for phase in "main" "auxiliary" ;
  do
    python3 main.py \
    --mode=infer \
    --configs=${configs_path}/first_stage/${model}.yaml \
    --experiments_path=${experiments_path}/first_stage/${model} \
    --img_path=${data_path}/${mode}/${phase}/img_crop \
    --output_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer \
    --pretrain_weight_path=${experiments_path}/first_stage/${model}/checkpoint/best_metric_model.pth || { echo 'Error during inference.'; exit 1; }
  done
done
#### 3. caculate metrics
## main test, GT is available in main phase
python3 caculate_metric.py \
--seg_path=${experiments_path}/first_stage/${model}/main_test_infer \
--label_path=${data_path}/test/main/label_crop_clip || { echo 'Error during caculating metrics.'; exit 1; }
fi
######################################################### information processing block ##################################################33##########
if [ ${IP} == "1" ]; then
### 1. fracture detection, fracture detection output is saved in the same folder as the input, named as
## args.root_path / f'{args.model}' / "fracture_detection" / f'{args.inference_stage}'
## like args.root_path/${model}/fracture_detection/args.inference_stage
for mode in "train" "test" ;
do
  for phase in "main" "auxiliary" ;
  do
    python3 post_processing/fracture_detection.py \
    --detection_stage=1 \
    --save_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_detection \
    --S_M=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer \
    --GT=${data_path}/${mode}/${phase}/label_crop_clip \
    --view
  done
done
## move the discontinuity label for training
for mode in "train" "test" ;
do
  for phase in "main" "auxiliary" ;
  do
    for postfix in "_sphere" "_sphere_GT" ;
    do
      echo /first_stage/${model}/${phase}_${mode}_infer_detection
      python3 post_processing/move_file.py \
      --data_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_detection \
      --save_path=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer${postfix} \
      --data_postfix=${postfix} || { echo 'Error during discontinuity detection.'; exit 1; }
    done
  done
done

#### 2. registration, registration output is saved in the same folder as the input, named as
### f"{args.net}/auxiliary_{args.mode}_infer_{args.reg_algorithm}_reg_{args.time}"
for mode in "train" "test" ;
do
  python3 registration/label_registration.py \
  --reg_algorithm=SyNRA \
  --time=1 \
  --save_root=${experiments_path}/first_stage/${model}/registration \
  --main_mask_path=${experiments_path}/first_stage/${model}/main_${mode}_infer \
  --auxiliary_mask_path=${experiments_path}/first_stage/${model}/auxiliary_${mode}_infer \
  --main_img_path=${data_path}/${mode}/main/img_crop \
  --auxiliary_img_path=${data_path}/${mode}/auxiliary/img_crop || { echo 'Error during registration.'; exit 1; }
done
fi
############################################################### refined segmentation #############################################################################
if [ ${RS} == "1" ]; then
if [ ${train} == "1" ]; then
# 1. train
python3 second_stage_main.py \
--mode=train \
--iters=50 \
--configs=${configs_path}/second_stage/${model}.yaml \
--experiments_path=${experiments_path}/second_stage/${model} \
--persist_path=${experiments_path}/second_stage/${model}/persist_cache \
--label_path=${data_path}/train/main/label_crop_clip \
--val_set=${data_path}/train/main/val.txt \
--train_set=${data_path}/train/main/train.txt \
--select_file=${experiments_path}/first_stage/${model}/main_train_infer_detection/cube_gt.txt \
--I_M=${data_path}/train/main/img_crop \
--CS_M=${experiments_path}/first_stage/${model}/main_train_infer \
--CS_DL=${experiments_path}/first_stage/${model}/main_train_infer_sphere \
--CS_DLGT=${experiments_path}/first_stage/${model}/main_train_infer_sphere_GT \
--I_A=${experiments_path}/first_stage/${model}/registration/auxiliary_train_img_SyNRA_reg_1 \
--CS_A=${experiments_path}/first_stage/${model}/registration/auxiliary_train_infer_SyNRA_reg_1 \
--CS_W=${experiments_path}/first_stage/${model}/checkpoint/best_metric_model.pth || { echo 'Error during training of refined segmentation.'; exit 1; }
fi
## 2. inference
## use test mode
for phase in "main" "auxiliary" ; # "main" "auxiliary" ;
do
  auxiliary_phase=""
  mode="test"
  if [ ${phase} == "main" ]; then
    auxiliary_phase="auxiliary"
  else auxiliary_phase="main"
  fi
  python3 second_stage_main.py \
  --mode=test \
  --configs=${configs_path}/second_stage/${model}.yaml \
  --experiments_path=${experiments_path}/second_stage/${model} \
  --output_path=${experiments_path}/second_stage/${model}/${phase}_${mode}_infer \
  --persist_path=${experiments_path}/second_stage/${model}/persist_cache \
  --label_path=${data_path}/${mode}/${phase}/label_crop_clip \
  --I_M=${data_path}/${mode}/${phase}/img_crop \
  --CS_M=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer \
  --CS_DL=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_sphere \
  --CS_DLGT=${experiments_path}/first_stage/${model}/${phase}_${mode}_infer_sphere_GT \
  --I_A=${experiments_path}/first_stage/${model}/registration/${auxiliary_phase}_${mode}_img_SyNRA_reg_1 \
  --CS_A=${experiments_path}/first_stage/${model}/registration/${auxiliary_phase}_${mode}_infer_SyNRA_reg_1 \
  --pretrain_weight_path=${experiments_path}/second_stage/${model}/checkpoint/best_metric_model.pth || { echo 'Error during inferring of refined segmentation.'; exit 1; }
done
fi
################################################################## inference block #########################################################################
### detection save direct merge result
### selected_detection save merge result which is selected the strategy that only successful RS will be selected to be merged
## 1 means discontinuity detection without GT within first stage
## 2 means discontinuity detection with GT within first stage
## postfix for searching: glob function will search for files with {postfix}.nii.gz, RCS and RCSGT might conflict, so we use _RCS
if [ ${IF} == "1" ]; then
## 1. second stage
for phase in "main" "auxiliary" ;
do
  python3 post_processing/fracture_detection.py \
  --detection_stage=2 \
  --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_1 \
  --S_M=${experiments_path}/second_stage/${model}/${phase}_test_infer \
  --S_M_postfix="_RCS" \
  --S_A_postfix="_CS_A" \
  --DL_postfix="_CS_DL" \
  --view || { echo 'Error during discontinuity detection.'; exit 1; }
  # move file to the same folder for better visualization
  for postfix in "_RS" "_RCS" "_RCSGT" "_CS_DL" "_CS_DLGT" "_GT" "_CS_M" "_CS_A";
  do
      python3 post_processing/move_file.py \
      --data_path=${experiments_path}/second_stage/${model}/${phase}_test_infer \
      --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_1 \
      --data_postfix=${postfix} \
      --save_postfix=${postfix} \
      --keep_parent || { echo 'Error during moving file.'; exit 1; }
  done
done
for phase in "main" "auxiliary" ;
do
  python3 post_processing/fracture_detection.py \
  --detection_stage=2 \
  --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_2 \
  --S_M=${experiments_path}/second_stage/${model}/${phase}_test_infer \
  --S_M_postfix="_RCSGT" \
  --S_A_postfix="_CS_A" \
  --DL_postfix="_CS_DLGT" \
  --view || { echo 'Error during discontinuity detection.'; exit 1; }
  declare -A save_postfix
  save_postfix=(["_ARCS"]="_ARCSGT" ["_AMASKED"]="_AMASKEDGT" ["_sphere"]="_sphere_2" ["_ARCS_TWO"]="_ARCS_TWOGT")
  for postfix in "_ARCS" "_AMASKED" "_ARCS_TWO" "_sphere";
  do
      python3 post_processing/move_file.py \
      --data_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_2 \
      --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_1 \
      --data_postfix=${postfix} \
      --save_postfix=${save_postfix[${postfix}]} \
      --keep_parent || { echo 'Error during moving file.'; exit 1; }
  done
done

#### select merge
for phase in "main" "auxiliary" ;
do
  python3 post_processing/fracture_detection.py \
  --detection_stage=2 \
  --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_selected_detection_1 \
  --S_M=${experiments_path}/second_stage/${model}/${phase}_test_infer \
  --S_M_postfix="_CS_M" \
  --S_A_postfix="_RS" \
  --DL_postfix="_CS_DL" \
  --view || { echo 'Error during selected discontinuity detection.'; exit 1; }
  # move file to the same folder for better visualization
  declare -A save_postfix
  save_postfix=(["_ARCS"]="_RCS_SELECTED" ["_AMASKED"]="_RCS_SELECTED_MASK" ["_sphere"]="_CS_DL_SELECTED")
  for postfix in "_ARCS" "_AMASKED" "_sphere";
  do
    python3 post_processing/move_file.py \
    --data_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_selected_detection_1 \
    --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_1 \
    --data_postfix=${postfix} \
    --save_postfix=${save_postfix[${postfix}]} \
    --keep_parent || { echo 'Error during moving file.'; exit 1; }
  done
done

for phase in "main" "auxiliary" ;
do
  python3 post_processing/fracture_detection.py \
  --detection_stage=2 \
  --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_selected_detection_2 \
  --S_M=${experiments_path}/second_stage/${model}/${phase}_test_infer \
  --S_M_postfix="_CS_M" \
  --S_A_postfix="_RS" \
  --DL_postfix="_CS_DLGT" \
  --view || { echo 'Error during selected discontinuity detection.'; exit 1; }
  # move file to the same folder for better visualization
  declare -A save_postfix
  save_postfix=(["_ARCS"]="_RCSGT_SELECTED" ["_AMASKED"]="_RCSGT_SELECTED_MASK" ["_sphere"]="_CS_DLGT_SELECTED")
  for postfix in "_ARCS" "_AMASKED" "_sphere" ;
  do
    python3 post_processing/move_file.py \
    --data_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_selected_detection_2 \
    --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_1 \
    --data_postfix=${postfix} \
    --save_postfix=${save_postfix[${postfix}]} \
    --keep_parent || { echo 'Error during moving file.'; exit 1; }
  done
done

## final merge
for phase in "main" "auxiliary" ;
do
  python3 post_processing/selected_final_merge.py \
  --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_1 \
  --save_postfix="_ARCS_SELECTED" \
  --data_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_1 \
  --rcs_selected_postfix="_RCS_SELECTED" \
  --arcs_postfix="_ARCS" \
  --dl1_postfix="_CS_DL" \
  --dl2_postfix="_sphere" || { echo 'Error during final merging.'; exit 1; }
  # move file to the same folder for better visualization
done

for phase in "main" "auxiliary" ;
do
  python3 post_processing/selected_final_merge.py \
  --save_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_1 \
  --save_postfix="_ARCSGT_SELECTED" \
  --data_path=${experiments_path}/second_stage/${model}/${phase}_test_infer_detection_1 \
  --rcs_selected_postfix="_RCSGT_SELECTED" \
  --arcs_postfix="_ARCSGT" \
  --dl1_postfix="_CS_DLGT" \
  --dl2_postfix="_sphere_2" || { echo 'Error during final merging.'; exit 1; }
  # move file to the same folder for better visualization
done

### moving file for visualization
########### Metric Calculation ##########
for postfix in "_RCS" "_RCSGT" "_ARCS" "_ARCSGT" "_ARCS_TWO" "_ARCS_TWOGT" "_CS_M" "_RCSGT_SELECTED" "_RCS_SELECTED" "_ARCS_SELECTED" "_ARCSGT_SELECTED" "_ARCS_SELECTED_TWO" "_ARCSGT_SELECTED_TWO";
do
  python3 caculate_metric.py \
  --seg_path=${experiments_path}/second_stage/${model}/main_test_infer_detection_1 \
  --seg_find=*/*${postfix}.nii.gz \
  --label_path=${data_path}/test/main/label_crop_clip \
  --metric_result_path=${experiments_path}/second_stage/${model}/main_test_infer_detection_1/${postfix}_metric.xlsx || { echo 'Error during calculating metric.'; exit 1; }
done
fi
