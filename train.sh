#!/bin/bash

# Train KWT on Speech commands v2 with 12 labels

KWS_PATH=$PWD
DATA_PATH=$KWS_PATH/data2
MODELS_PATH=$KWS_PATH/models_data_v2_12_labels
CMD_TRAIN="python -m kws_streaming.train.model_train_eval"
TRAIN_DIR="$MODELS_PATH/comet_$(date +%Y%m%d_%H%M%S)_$$"

$CMD_TRAIN \
--data_url '' \
--data_dir $DATA_PATH/ \
--train_dir "$TRAIN_DIR/" \
--mel_upper_edge_hertz 7600 \
--optimizer 'adamw' \
--lr_schedule 'linear' \
--train 1 \
--how_many_training_steps '20000' \
--eval_step_interval 500 \
--warmup_epochs 0 \
--l2_weight_decay 0.1 \
--learning_rate '0.001' \
--batch_size 8 \
--label_smoothing 0.1 \
--clip_duration_ms 2000 \
--window_size_ms 25.0 \
--window_stride_ms 10.0 \
--mel_num_bins 40 \
--dct_num_features 0 \
--resample 0.15 \
--alsologtostderr \
--train 1 \
--use_spec_augment 1 \
--time_masks_number 2 \
--time_mask_max_size 25 \
--frequency_masks_number 2 \
--frequency_mask_max_size 7 \
--pick_deterministically 1 \
--wanted_words 'yes,no' \
kws_transformer \
--num_layers 2 \
--heads 4 \
--d_model 32 \
--mlp_dim 64 \
--dropout1 0. \
--attention_type 'patch' \
--patch_size '4,40' \
--pooling mean \
--prenorm 0 # if False, use postnorm
