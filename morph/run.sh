#!/bin/bash
# Load .env if present (MORPH_REPO_DIR, MORPH_DATA_ROOT, MORPH_RESULT_DIR)
if [ -f .env ]; then set -a; source .env; set +a; fi

# Outputs and checkpoints: use MORPH_RESULT_DIR from .env or set here
result_dir="${MORPH_RESULT_DIR:-}"

# Running models -------------------------------------------------------------
random_seeds=(12)
modality="rna" # "rna" or "ops"
model="MORPH" # choose model type: ("MOE" or "MOE_no_residual1" or "MOE_moe_3expert"), if use moe_3expert, please specify the other two priors in label_2 and label_3
dataset_name="replogie_RPE1_essential"
leave_out_test_set_ids=("heldout_perts")
representation_types=("DepMap_GeneEffect")
# K562_essential_norm: full genome; run HVG in loader
use_hvg_flag="--use_hvg"
n_top_genes=5000
label_2="None" #"GenePT_v1", None
label_3="None" #"STRING", None
null_label='zeros' # 'gaussian_normalized' or 'gaussian' or 'zeros'
epochs=100
tolerance_epochs=20 #set to 20 by default
batch_size=32
lr=1e-4
MMD_sigma=1500
mxAlpha=1
mxBeta=2
Gamma1=0.5 
Gamma2=1
latdim_ctrl=50
latdim_ptb=50
geneset_num=50
geneset_dim=50
device="cuda:0"

model_name_list=("best_model.pt" "best_model_val.pt")

# Running models and evaluations ----------------------------------------------
for test_set_name in "${leave_out_test_set_ids[@]}"
do
    for representation_type in "${representation_types[@]}"
    do
        for random_seed in "${random_seeds[@]}"
        do
            echo "Running models with test_set_name=$test_set_name, dataset_name=$dataset_name, random_seed=$random_seed"
            python run.py --modality "$modality" \
                          --random_seed "$random_seed" \
                          --dataset_name "$dataset_name" \
                          --leave_out_test_set_id "$test_set_name" \
                          --device "$device" \
                          --model "$model" \
                          --label "$representation_type" \
                          --label_2 "$label_2" \
                          --label_3 "$label_3" \
                          --null_label "$null_label" \
                          --epochs "$epochs" \
                          --tolerance_epochs "$tolerance_epochs" \
                          --batch_size "$batch_size" \
                          --lr "$lr" \
                          --MMD_sigma "$MMD_sigma" \
                          --mxAlpha "$mxAlpha" \
                          --mxBeta "$mxBeta" \
                          --Gamma1 "$Gamma1" \
                          --Gamma2 "$Gamma2" \
                          --latdim_ctrl "$latdim_ctrl" \
                          --latdim_ptb "$latdim_ptb" \
                          --geneset_num "$geneset_num" \
                          --geneset_dim "$geneset_dim" \
                          ${use_hvg_flag:+$use_hvg_flag --n_top_genes "$n_top_genes"} \
                          ${result_dir:+--result_dir "$result_dir"}

            echo "Running evaluations with test_set_name=$test_set_name, dataset_name=$dataset_name, random_seed=$random_seed"
            for model_name in "${model_name_list[@]}"
            do
                python run_eval_best_val.py --modality "$modality" \
                                            --dataset_name "$dataset_name" \
                                            --leave_out_test_set_id "$test_set_name" \
                                            --label "$representation_type" \
                                            --model_type "$model" \
                                            --model_name "$model_name" \
                                            --device "$device" \
                                            ${result_dir:+--result_dir "$result_dir"}
            done
        done
    done
done