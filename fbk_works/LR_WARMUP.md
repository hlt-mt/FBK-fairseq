# The Warmup Dilemma: How Learning Rate Strategies Impact Speech-to-Text Model Convergence (IWSLT 2025)

Code and models for the paper "[The Warmup Dilemma: How Learning Rate Strategies Impact Speech-to-Text Model Convergence](https://arxiv.org/abs/2505.23420)"
published at IWSLT 2025.

## 📌 Models and Logs

The artifacts resulting from our experiments can be downloaded from the links below.

 - Piecewise LR Warmup: [checkpoints](https://fbk-my.sharepoint.com/:f:/g/personal/mgaido_fbk_eu/EmXZzDENjpFOnWNxPNsJJGkBiCYdOZ1RBnbVSSrSSnir9w?e=XAEpsr) and [log](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EZMREkWlYMdHlY7GAzFNTSEB1ysdgvcIntSKmfeYC84qPQ?e=MqGrev)
 - Exponential LR Warmup: [checkpoints](https://fbk-my.sharepoint.com/:f:/g/personal/mgaido_fbk_eu/EvKWXjh1K3dAtLe_M48osJMBY3xNp_iaPBDgy4IBMSP5ZQ?e=Zaa2VU) and [log](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EcGn9NEqj9dBlwsG_xNPAckBWJxqWCU_Hd-VPOBzlq2l-g?e=isTMKn)
 - Polynomial LR Warmup: [checkpoints](https://fbk-my.sharepoint.com/:f:/g/personal/mgaido_fbk_eu/EvFQzMxZHmxFiw4EzMe5k9wB_wC-07vCXRVVmvsHKLfbzA?e=NLzCh8) and [log](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EaUGoNq4Ks5Pmccmzlo8PGcBd_Pa6S6jxRJcIUc76O27yQ?e=aJLQGz)

Config YAML: [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EWIna6K_O3NMu1KjFd4-B24B2KFzYZNFjVkIJVeZgeCrTg?e=lXnp3K)

Sentencepiece vocabulary: [model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXA96m-EcCxEp72iaaLnKYYBCJGi0sPwKuHHzevC9hBROA?e=UBdMFv) and [txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EY_8oc4a5JZJjn7Pf7zvkJgB0ErvPVSugHqIoUX-1zRWAA?e=qsYFpU)

## Training Scripts

To replicate the experiments mentioned in the paper, after downloading and preprocessing the 
[FAMA training data available on HuggingFace](https://huggingface.co/datasets/FBK-MT/fama-data),
use the following scripts.

### Piecewise LR scheduler

To use the piecewise learning rate scheduler, execute the following script:

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
    --optimizer adam --lr $ETA --intermediate-lr $ETA_INTERMEDIATE --lr-scheduler piecewise_warmup \
    --weight-decay 0.001 \
    --clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
    --seed 1 --skip-invalid-size-inputs-valid-test --max-source-positions 3000 --max-target-positions 512 \
    --update-freq 5 --max-tokens 5500 --num-workers 1 \
    --max-update 1000000 --batch-unsafe-relative-shift \
    --keep-interval-updates 25 --validate-interval 1000 --save-interval-updates 1000 \
    --log-format simple --distributed-port 61024 >> $SAVEDIR/train.log 2> $SAVEDIR/train.err
```

where `ETA=2e-4` and `ETA_INTERMEDIATE=2e-5`. `CONFIG` is a YAML file provided in the previous section.

### Exponential LR scheduler

For the exponential LR scheduler, use:

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
    --optimizer adam --lr $ETA --lr-scheduler exponential_warmup --alpha-lr $ALPHA \
    --weight-decay 0.001 \
    --clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
    --seed 1 --skip-invalid-size-inputs-valid-test --max-source-positions 3000 --max-target-positions 512 \
    --update-freq 5 --max-tokens 5500 --num-workers 1 \
    --max-update 1000000 --batch-unsafe-relative-shift \
    --keep-interval-updates 25 --validate-interval 1000 --save-interval-updates 1000 \
    --log-format simple --distributed-port 61024 >> $SAVEDIR/train.log 2> $SAVEDIR/train.err
```

where `ETA` is the same as before and `ALPHA` is set to 1.5 in the experiment reported in the main paper.

### Polynomial LR scheduler

For the polynomial LR scheduler, use:

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
    --optimizer adam --lr $ETA --lr-scheduler polynomial_warmup --power-lr-warmup $ALPHA \
    --weight-decay 0.001 \
    --clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
    --seed 1 --skip-invalid-size-inputs-valid-test --max-source-positions 3000 --max-target-positions 512 \
    --update-freq 5 --max-tokens 5500 --num-workers 1 \
    --max-update 1000000 --batch-unsafe-relative-shift \
    --keep-interval-updates 25 --validate-interval 1000 --save-interval-updates 1000 \
    --log-format simple --distributed-port 61024 >> $SAVEDIR/train.log 2> $SAVEDIR/train.err
```
where `ETA` and `ALPHA` have the same value of the exponential LR scheduler.

## Generate

To reproduce our results, you can use the above models and generate the output using the following script:

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


## ⭐ Citation
If you use this work, please cite:
```bibtex
@inproceedings{gaido-et-al-2025-warmup,
  title = {{The Warmup Dilemma: How Learning Rate Strategies Impact Speech-to-Text Model Convergence}},
  author = {Gaido, Marco and Papi, Sara and Bentivogli, Luisa and Brutti, Alessio and Cettolo, Mauro and Gretter, Roberto and Matassoni, Marco and Nabih, Mohamed and Negri, Matteo},
  booktitle = "Proceedings of the 22nd International Conference on Spoken Language Translation (IWSLT)",
  year = "2025",
  address = "Vienna, Austria",
}
```