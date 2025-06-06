# FAMA: The First Large-Scale Open-Science Speech Foundation Model for English🇬🇧 and Italian🇮🇹

<img src="https://huggingface.co/FBK-MT/fama-medium-asr/resolve/main/FAMA.png" alt="FAMA" width="100%">

Code for "[FAMA: The First Large-Scale Open-Science Speech Foundation Model for English and Italian](https://arxiv.org/abs/2505.22759)".

## 🚩 Models and Logs

### 🤖 Fairseq models

All artifacts resulting from our Fairseq experiments can be downloaded from the links below:

- **FAMA-medium (878M)**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EcjIhgMd6D1EuDDmH7xv5CcBQ9lqSYEip4nr3vawbVn8vA) - [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EWIna6K_O3NMu1KjFd4-B24B2KFzYZNFjVkIJVeZgeCrTg?e=lXnp3K) - [spm_en_it_16000_lid.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXA96m-EcCxEp72iaaLnKYYBCJGi0sPwKuHHzevC9hBROA?e=UBdMFv) - [spm_en_it_16000_lid.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EY_8oc4a5JZJjn7Pf7zvkJgB0ErvPVSugHqIoUX-1zRWAA?e=qsYFpU) 
- **FAMA-small (479M)**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EcGimcxhq99HkBLKcDjXBcABe4E4V4EOtzvqN3r5btrGzQ) - [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EWIna6K_O3NMu1KjFd4-B24B2KFzYZNFjVkIJVeZgeCrTg?e=lXnp3K) - [spm_en_it_16000_lid.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXA96m-EcCxEp72iaaLnKYYBCJGi0sPwKuHHzevC9hBROA?e=UBdMFv) - [spm_en_it_16000_lid.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EY_8oc4a5JZJjn7Pf7zvkJgB0ErvPVSugHqIoUX-1zRWAA?e=qsYFpU) 
- **FAMA-medium-asr (878M)**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EdiVitSStwhMo67L8Eos8z4Bo04ubQvSHS_y2JeY1A3wpA) - [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EWIna6K_O3NMu1KjFd4-B24B2KFzYZNFjVkIJVeZgeCrTg?e=lXnp3K) - [spm_en_it_16000_lid.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXA96m-EcCxEp72iaaLnKYYBCJGi0sPwKuHHzevC9hBROA?e=UBdMFv) - [spm_en_it_16000_lid.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EY_8oc4a5JZJjn7Pf7zvkJgB0ErvPVSugHqIoUX-1zRWAA?e=qsYFpU) 
- **FAMA-small-asr (878M)**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EeZc07RzhgZJoNUztMH6DtkBQjgE4lZZyfnp28EVWuq12A) - [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EWIna6K_O3NMu1KjFd4-B24B2KFzYZNFjVkIJVeZgeCrTg?e=lXnp3K) - [spm_en_it_16000_lid.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXA96m-EcCxEp72iaaLnKYYBCJGi0sPwKuHHzevC9hBROA?e=UBdMFv) - [spm_en_it_16000_lid.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EY_8oc4a5JZJjn7Pf7zvkJgB0ErvPVSugHqIoUX-1zRWAA?e=qsYFpU) 

### 🤗 HuggingFace models

The models are also available on HuggingFace transformers:

- **FAMA-medium (878M)**: 🔗[https://hf.co/FBK-MT/fama-medium](https://hf.co/FBK-MT/fama-medium)
- **FAMA-small (479M)**: 🔗[https://hf.co/FBK-MT/fama-small](https://hf.co/FBK-MT/fama-small)
- **FAMA-medium-asr (878M)**: 🔗[https://hf.co/FBK-MT/fama-medium-asr](https://hf.co/FBK-MT/fama-medium-asr)
- **FAMA-small-asr (878M)**: 🔗[https://hf.co/FBK-MT/fama-small-asr](https://hf.co/FBK-MT/fama-small-asr)

## 📌 Training

The training is performed in two steps, ASR pretraining, and ASR + ST training. 
**FAMA-medium-asr** and **FAMA-small-asr** are the resulting checkpoints obtained after the first step.

To replicate the experiments mentioned in the paper, after downloading and preprocessing the 
[FAMA training data available on HuggingFace 🤗](https://huggingface.co/datasets/FBK-MT/fama-data) in the `$DATAROOT`,
use the following scripts. The training is performed using 16 NVIDIA A100 (64GB) GPUs.


### 📣 FAMA medium

For the medium model training, we use [the piecewise learning rate scheduler](https://arxiv.org/abs/2505.23420).

#### Step 1️⃣
Run the following script:
```bash
python fbk-fairseq/train.py $DATAROOT \
    --train-subset train_commonvoice-en,train_commonvoice-it,train_covost2-en,train_covost2-it,train_fleurs-en,train_fleurs-it,train_librilightlarge-en,train_librilightmedium-en,train_librilightsmall-en,train_librispeech-en,train_mls-en,train_mls-it,train_voxpopuliasr-en,train_voxpopuliasr-it,train_voxpopuli-en,train_voxpopuli-it,train_youtubecommons-en_filtered,train_youtubecommons-it \
    --valid-subset dev_commonvoice-en,dev_commonvoice-it --config-yaml $CONFIG \
    --save-dir $SAVEDIR --user-dir fbk-fairseq/examples/speech_to_text \
    --encoder-layers 24 --decoder-layers 12 \
    --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --task speech_to_text_ctc --criterion ctc_multi_loss --underlying-criterion joint_cross_entropy_ctc \
    --label-smoothing 0.1 --ctc-encoder-layer 16 --ctc-weight 1.0 --primary-loss-weight 5.0 --auxiliary-loss-weight 2.0 \
    --arch conformer_joint_ctc --ignore-prefix-size 1 \
    --optimizer adam \
    --lr 2e-4 --intermediate-lr 2e-5 --lr-scheduler piecewise_warmup --warmup-updates 50000 --intermediate-warmup-updates 25000 \
    --weight-decay 0.001 \
    --clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
    --seed 1 --skip-invalid-size-inputs-valid-test --max-source-positions 3000 --max-target-positions 512 \
    --update-freq 5 --max-tokens 5500 --num-workers 1 \
    --max-update 1000000 --batch-unsafe-relative-shift \
    --keep-interval-updates 25 --validate-interval 1000 --save-interval-updates 1000 \
    --log-format simple --distributed-port 61024 >> $SAVEDIR/train.log 2> $SAVEDIR/train.err
```

where `$SAVEDIR` is the path to the folder where checkpoints and logs of Step 1️⃣ are saved. 
`CONFIG` is a YAML file provided in the previous section.

#### Step 2️⃣
Average the last 25 checkpoints obtained from the previous step:
```sh
python ${FAIRSEQ_DIR}/scripts/average_checkpoints.py \
  --inputs ${SAVEDIR} \
  --output "${SAVEDIR}/avg25.pt" \
  --num-epoch-checkpoints 25
```

and then run:
```bash
python fbk-fairseq/train.py $DATAROOT \
        --train-subset train_st_librispeech-en_it,train_st_commonvoice-en_it,train_st_commonvoice-it_en,train_st_covost2-en_it,train_st_covost2-it_en,train_st_covost2-it_en_gold,train_st_fleurs-en_it,train_st_fleurs-it_en,train_st_librilightlarge-en_it,train_st_librilightmedium-en_it,train_st_librilightsmall-en_it,train_st_mls-en_it,train_st_mls-it_en,train_st_voxpopuliasr-en_it,train_st_voxpopuliasr-it_en,train_st_voxpopuli-en_it,train_st_voxpopuli-it_en,train_st_youtubecommons-en_it,train_st_youtubecommons-it_en \
        --valid-subset dev_commonvoice-en,dev_commonvoice-it,dev_st_covost2-it_en,dev_st_fleurs-en_it --config-yaml $CONFIG \
        --save-dir $STEP2SAVEDIR --user-dir examples/speech_to_text \
        --encoder-layers 24 --decoder-layers 12 \
        --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 --decoder-attention-heads 16 \
        --task speech_to_text_ctc_asr_st --criterion ctc_multi_loss --underlying-criterion joint_cross_entropy_ctc \
        --label-smoothing 0.1 --ctc-encoder-layer 16 --ctc-weight 1.0 --primary-loss-weight 5.0 --auxiliary-loss-weight 2.0 \
        --arch conformer_joint_ctc --ignore-prefix-size 1 --p-sampling-asr 0.5 \
        --optimizer adam --lr 1e-5 --lr-scheduler fixed \
        --weight-decay 0.001 \
        --clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
        --seed 1 --skip-invalid-size-inputs-valid-test \
        --update-freq 6 --max-tokens 4500 --num-workers 1 \
        --max-update 1000000 --batch-unsafe-relative-shift \
        --finetune-from-model $SAVEDIR/avg25.pt --max-source-positions 3000 --max-target-positions 512 \
        --keep-interval-updates 25 --validate-interval 1000 --save-interval-updates 1000 \
        --log-format simple --distributed-port 61025 >> $STEP2SAVEDIR/train.log 2>> $STEP2SAVEDIR/train.err
```

where `$STEP2SAVEDIR` is the path to the folder where checkpoints and logs of Step 2️⃣ are saved. 
`CONFIG` is the same YAML file, provided in the previous section.

### 📣 FAMA small

#### Step 1️⃣
Run the following script:
```bash
python fbk-fairseq/train.py $DATAROOT \
    --train-subset train_commonvoice-en,train_commonvoice-it,train_covost2-en,train_covost2-it,train_fleurs-en,train_fleurs-it,train_librilightlarge-en,train_librilightmedium-en,train_librilightsmall-en,train_librispeech-en,train_mls-en,train_mls-it,train_voxpopuliasr-en,train_voxpopuliasr-it,train_voxpopuli-en,train_voxpopuli-it,train_youtubecommons-en_filtered,train_youtubecommons-it \
    --valid-subset dev_commonvoice-en,dev_commonvoice-it --config-yaml $CONFIG \
    --save-dir $SAVEDIR --user-dir fbk-fairseq/examples/speech_to_text \
    --encoder-layers 12 --decoder-layers 6 \
    --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 --decoder-attention-heads 16 \
    --task speech_to_text_ctc --criterion ctc_multi_loss --underlying-criterion joint_cross_entropy_ctc \
    --label-smoothing 0.1 --ctc-encoder-layer 8 --ctc-weight 1.0 --primary-loss-weight 5.0 --auxiliary-loss-weight 2.0 \
    --arch conformer_joint_ctc --ignore-prefix-size 1 \
    --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 25000 \
    --weight-decay 0.001 \
    --clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
    --seed 1 --skip-invalid-size-inputs-valid-test \
    --update-freq 2 --max-tokens 10000 --num-workers 1 \
    --max-update 1000000 --batch-unsafe-relative-shift \
    --keep-interval-updates 25 --validate-interval 1000 --save-interval-updates 1000 \
    --log-format simple --distributed-port 61024 >> $SAVEDIR/train.log 2> $SAVEDIR/train.err
```

where `$SAVEDIR` is the path to the folder where checkpoints and logs of Step 1️⃣ are saved. 
`CONFIG` is a YAML file provided in the previous section. 

#### Step 2️⃣
Average the last 25 checkpoints obtained from the previous step:
```sh
python ${FAIRSEQ_DIR}/scripts/average_checkpoints.py \
  --inputs ${SAVEDIR} \
  --output "${SAVEDIR}/avg25.pt" \
  --num-epoch-checkpoints 25
```

and then run:
```bash
python fbk-fairseq/train.py $DATAROOT \
        --train-subset train_st_librispeech-en_it,train_st_commonvoice-en_it,train_st_commonvoice-it_en,train_st_covost2-en_it,train_st_covost2-it_en,train_st_covost2-it_en_gold,train_st_fleurs-en_it,train_st_fleurs-it_en,train_st_librilightlarge-en_it,train_st_librilightmedium-en_it,train_st_librilightsmall-en_it,train_st_mls-en_it,train_st_mls-it_en,train_st_voxpopuliasr-en_it,train_st_voxpopuliasr-it_en,train_st_voxpopuli-en_it,train_st_voxpopuli-it_en,train_st_youtubecommons-en_it,train_st_youtubecommons-it_en \
        --valid-subset dev_commonvoice-en,dev_commonvoice-it,dev_st_covost2-it_en,dev_st_fleurs-en_it --config-yaml $CONFIG \
        --save-dir $STEP2SAVEDIR --user-dir examples/speech_to_text \
        --encoder-layers 12 --decoder-layers 6 \
        --encoder-embed-dim 1024 --encoder-ffn-embed-dim 4096 --encoder-attention-heads 16 --decoder-attention-heads 16 \
        --task speech_to_text_ctc_asr_st --criterion ctc_multi_loss --underlying-criterion joint_cross_entropy_ctc \
        --label-smoothing 0.1 --ctc-encoder-layer 8 --ctc-weight 1.0 --primary-loss-weight 5.0 --auxiliary-loss-weight 2.0 \
        --arch conformer_joint_ctc --ignore-prefix-size 1 --p-sampling-asr 0.5 \
        --optimizer adam --lr 1e-4 --lr-scheduler fixed \
        --weight-decay 0.001 \
        --clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
        --seed 1 --skip-invalid-size-inputs-valid-test \
        --update-freq 2 --max-tokens 10000 --num-workers 1 \
        --max-update 1000000 --batch-unsafe-relative-shift \
        --finetune-from-model $SAVEDIR/avg25.pt --max-source-positions 3000 --max-target-positions 512 \
        --keep-interval-updates 25 --validate-interval 1000 --save-interval-updates 1000 \
        --log-format simple --distributed-port 61025 >> $STEP2SAVEDIR/train.log 2>> $STEP2SAVEDIR/train.err
```

where `$STEP2SAVEDIR` is the path to the folder where checkpoints and logs of Step 2️⃣ are saved. 
`CONFIG` is the same YAML file, provided in the previous section.

## 📌 Generate

To obtain the final checkpoint, average the last 25 checkpoints obtained from Step 2️⃣, using [the same script reported above](#step-2).

To perform a standard generate (no joint CTC rescoring), use the following script:

```bash
python fbk-fairseq/fairseq_cli/generate.py $DATA_ROOT \
    --user-dir fbk-fairseq/examples/speech_to_text/ --config-yaml $CONFIG_YAML \
    --gen-subset $SPLIT --prefix-size 1 \
    --max-tokens 80000 --unkpen 10000 --beam 5 \
    --max-source-positions 12000 --max-target-positions 4000 \
    --model-overrides "{'max_source_positions':12000,'max_target_positions':4000, 'batch_unsafe_relative_shift': False}" \
    --task speech_to_text_ctc --criterion ctc_multi_loss \
    --underlying-criterion label_smoothed_cross_entropy --no-repeat-ngram-size 5 \
    --path $MODEL > output.log
```
where `SPLIT` is set to the name of the TSV file in `DATA_ROOT` which contains the test data,
`MODEL` is the path to the model checkpoint and `CONFIG_YAML` is the path to the YAML config file,
which is the same used for training.

To perform a generate with joint CTC rescoring, use the following script:
```bash
python fbk-fairseq/fairseq_cli/generate.py $DATA_ROOT \
    --user-dir fbk-fairseq/examples/speech_to_text/ --config-yaml $CONFIG_YAML \
    --gen-subset $SPLIT --prefix-size 1 \
    --max-tokens 80000 --unkpen 10000 --beam 5 \
    --max-source-positions 12000 --max-target-positions 4000 \
    --model-overrides "{'max_source_positions':12000,'max_target_positions':4000, 'batch_unsafe_relative_shift': False}" \
    --task speech_to_text_joint_decoding --ctc-decode-weight 0.2 --criterion ctc_multi_loss \
    --underlying-criterion joint_cross_entropy_ctc --no-repeat-ngram-size 5 \
    --path $MODEL > output.log
```


## ⭐ Citation
If you use this work, please cite:
```bibtex
@misc{papi2025famalargescaleopensciencespeech,
      title={FAMA: The First Large-Scale Open-Science Speech Foundation Model for English and Italian}, 
      author={Sara Papi and Marco Gaido and Luisa Bentivogli and Alessio Brutti and Mauro Cettolo and Roberto Gretter and Marco Matassoni and Mohamed Nabih and Matteo Negri},
      year={2025},
      eprint={2505.22759},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22759}, 
}
```
