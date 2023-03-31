# Correctness of Conformer implementation

This README contains the instructions to replicate the training and evaluation of the models in the paper
[Reproducibility is Nothing Without Correctness: The Importance of Testing Code in NLP](https://arxiv.org/abs/2303.16166).
In addition, we release the pre-trained models used in the paper.


## Setup
Clone this repository and install it as explained in the original [Fairseq(-py)](https://github.com/pytorch/fairseq).
For the experiments we used MuST-C, make sure to [download the corpus](https://ict.fbk.eu/must-c/).
Follow the [preprocessing steps of Speechformer](SPEECHFORMER.md#preprocessing) to preprocess the MuST-C data.

## Training
The **bug-free** version of the Conformer-based model can be trained by passing the target language (`LANG`), the folder containing the 
MuST-C preprocessed data (`MUSTC_ROOT`), the task in `TASK` (either `asr` or `st`), and the directory in
which the checkpoints and training log will be saved (`SAVE_DIR`).
```bash
LANG=$1
MUSTC_ROOT=$2
TASK=$3
SAVE_DIR=$4

mkdir -p $SAVE_DIR

python ${FBK_fairseq}/train.py ${MUSTC_ROOT} \
        --train-subset train_${TASK}_src --valid-subset dev_${TASK}_src \
        --user-dir examples/speech_to_text --seed 1 \
        --num-workers 1 --max-update 100000 --patience 10 --keep-last-epochs 12 \
        --max-tokens 40000 --update-freq 4 \
        --task speech_to_text_ctc --config-yaml config.yaml  \
        --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch conformer \
        --ctc-encoder-layer 8 --ctc-weight 0.5  --ctc-compress-strategy avg \
        --optimizer adam --adam-betas '(0.9, 0.98)' \
        --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 25000 \
        --clip-norm 10.0 \
        --skip-invalid-size-inputs-valid-test \
        --save-dir ${SAVE_DIR} \
        --log-format simple > $SAVE_DIR/train.log 2> $SAVE_DIR/train.err

python ${FBK_fairseq}/scripts/average_checkpoints.py \
        --input $SAVE_DIR --num-epoch-checkpoints 5 \
        --checkpoint-upper-bound $(ls $SAVE_DIR | head -n 5 | tail -n 1 | grep -o "[0-9]*") \
        --output $SAVE_DIR/avg5.pt

if [ -f $SAVE_DIR/avg5.pt ]; then
  rm $SAVE_DIR/checkpoint??.pt
fi
```
The script will train the model and make the average of 5 checkpoints (best, 2 preceedings and 2 succeedings).

To remove the CTC Compression from the model, remove `--ctc-compress-strategy avg` from the script.

Due to its increased training time, the bug-fix of the padding problem present in the relative positional encodings (🪲3)
can be disabled by adding `--batch-unsafe-relative-shift` to the script.

To remove the bug-fix relative to the Convolution Module (🪲1) revert the commit: 
`[!63][CONFORMER] Correction to the Convolutional Layer of Conformer for missing padding`.

To remove the bug-fix relative to the Initial Subsampling (🪲2) rever the commit:
`[!69][TRANSFORMER][CONFORMER][BUG] Fix padding in initial convolutional layers`.

To enable or disable TF32 you need to respectively set to `True` or `False`
the flags `torch.backends.cuda.matmul.allow_tf32` and `torch.backends.cudnn.allow_tf32`.
Please notice that their default value depends on the version of PyTorch you are using
(our experiments have been carried out with PyTorch 1.11, where they are enabled by default),
and they have effect only on Ampere GPU (if you are using a V100 GPU, for instance, TF32 cannot
be enabled). For more information, refer to the 
[official pytorch page](https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices).

## Evaluation
Generate the output by varying the batch size `SENTENCES={1, 10, 100}` and check its independence from it:
the generated files must be the same.
```bash
python ${FBK_fairseq}/fairseq_cli/generate.py ${MUSTC_ROOT} \
        --user-dir examples/speech_to_text \
        --config-yaml config.yaml --gen-subset tst-COMMON_st_src \
        --max-sentences ${SENTENCES} \
        --max-source-positions 10000 --max-target-positions 1000 \
        --task speech_to_text_ctc \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
        --beam 5 --no-repeat-ngram-size 5 --path ${SAVE_DIR}/avg5.pt > ${SAVE_DIR}/tst-COMMON.${SENTENCES}.out
```


## Pretrained models
### Common files:
- Source dictionary SentencePiece model and fairseq dictionary: 
[srcdict.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EdAgeZdaw5BEjv6PUPEycvoBZHeOMqZ69ciEAIHM0XoBbw?e=t2z5G1),
[srcdict.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EY6_YCFCDjxBlBvm2_8UQFEB9ehLmFoLiGj2r7GGe_pL0A?e=NhIhkz)
- Target dictionary SentencePiece model and fairseq dictionary:
  - **en (ASR)**: same as srcdict.model and srcdict.txt 
  - **en-de**: 
  [tgtdict.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Eamb-6DsnklHq-4CZOZA9nYBKZ0XXnz0UdeOb49UXYlLVQ?e=yroKIk),
  [tgtdict.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EVOJ0yFgZZpEqvHUlzhjqOEBkV7U26iryO-bpobz_5q_fQ?e=i2gdi0)
  - **en-es**:
  [tgtdict.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EWmh3csXbEVPmBSI7xeemVMBHqlSEDJHl3JmUOXzPRwCAA?e=T53pKl),
  [tgtdict.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EduV9z-HroFOgh2xQjhdShIBmCs-6PmvgqkzPfcQmXsXdQ?e=iehKch)
  - **en-fr**:
  [tgtdict.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXQfn6DYxC1CskMO7lJMaxIB23Wa4xIWOtsX2SIukOOM9A?e=HyvZrB),
  [tgtdict.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/ETV367Z8xJ1Egz9E_cKBdykB9iYgDdEj1xLKBLRTANWCUA?e=Y5CUky)
  - **en-it**: 
  [tgtdict.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EX_w-V-SN1dLkEEJWrXbK_UBxHQL0zJaJuzIM_ZzosICmg?e=Wf0VKk),
  [tgtdict.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/ERAhMZjPoJNHkPWih7v0GfoBus4jG0WD3XPRmK5CgaV3wA?e=lG50Ny)
  - **en-nl**: 
  [tgtdict.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EZ8C2AySmHxLi7qDcf4PcvEBEg5tkVXK9jsB1t8v0F3Maw?e=6VCiwb),
  [tgtdict.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EWvoJ9Lb97RGqaUaFgsWPlMBYgo9uTIxUUY6KidHnZErhw?e=986D7S)
  - **en-pt**:
  [tgtdict.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EX9u-0PII8JKpnNensFj5ygBqVZrcPYoE8RWC8VryspzTg?e=2LjDH5),
  [tgtdict.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EZ2TMRgLtudCuvXcsjCzOtkBjWVSdsof1LGmt9bOtQn9gg?e=boCBtQ)
  - **en-ro**:
  [tgtdict.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Ec_zzPD3sTtCkNmibsMUUQUBWQHxinzoNvSRCCx6c_JhzA?e=Q5pDs7),
  [tgtdict.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EbkE3WFxh4lDiR7aB9wA6NoBaQIZnM6MnWscLKD-h5nMTw?e=QgoD95)
  - **en-ru**:
  [tgtdict.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EbIHTXtOjh5PsUgGmex_-osBxxeSwGgYxtF_tbJLFuxuZA?e=CocdUW),
  [tgtdict.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/ET3q0JaCqDFMvbuWc5RjIP0BRl1hhl-noNO-jWdHqdAMkw?e=wPYf9Q)
- config yaml: 
```bash
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: tgtdict.model
bpe_tokenizer_src:
  bpe: sentencepiece
  sentencepiece_model: srcdict.model
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
specaugment:
  freq_mask_F: 27
  freq_mask_N: 1
  time_mask_N: 1
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - utterance_cmvn
  _train:
  - utterance_cmvn
  - specaugment
vocab_filename: tgtdict.txt
vocab_filename_src: srcdict.txt
```
### Checkpoints
| Code | Model             | en (ASR)   | en-de      | en-es      | en-fr | en-it | en-nl | en-pt | en-ro | en-ru |
|---|-------------------|------------|------------|------------|-------|-------|-------|-------|-------|-------|
| ✅ | Conformer         | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EUqksBx2Ay5KvNc5skqC1h0BEUqehOmFK5crI9-t0cjp2g?e=kLDFlB) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EQwB8XIZTRJAqcAzQt7HDYEBFXXyUnbGxsQoMlObUE_VYw?e=PibCgl) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ES7PfW59F7xCiBh8XkvS3zEBnXlflJO4c32GIqIIyRiMqg?e=66lNIy) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EetLvWaVnwFCmikWSnMGFmgBt32mN6vIpNIEmZPAe1s-Xg?e=3thQGt) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EQRe9MSoHYlBoEslKOL4IDABCamEBZRa9gHvKtnsiGpa6Q?e=enMpNr) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXJ3372oAkNKgvbVRKJQYU4BHSL0alWpnGVh8cFVGgh-vQ?e=adZdFw) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ESe_qUZHNJ1FgvCOF-3rZNIB4KEJyST42G-2XU7sq4GDOg?e=u6IRIR) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Eb9zGA7aYJ1DtfvTyuqadkgBrDp2DSaVbQvpgXWLki4kFQ?e=Lqc6qY) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Eargi0N-1lBFjL4PNmBj_xwB2vqGTOTqILQpdvierlzfnQ?e=moBeMR) |
| ✅ | + CTC Compression | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ESmyt1nvd_xMo0w8KjI6sosB4rxOo5TvFE4RXA6FNYfHcA?e=K6OOEH) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXviVZLI1dZKjmHUsAKBcacBiI_SrI2ukUwg3eo4-hk6mg?e=QwYk3n) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ET84cXmoijpCmgyt-lxbgbcBYbSm_Ld6d8lsiMjllF0QCg?e=KBOpGZ) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EdpmlfVfzI1Cki8jSYrVT1MBB8YYk5DYtlFfcXfYtbdliQ?e=YU94SL) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EUbzNVqNMw1DmjoAGA-FG4kBIIkzLJADMaC1QOTcG5ydzA?e=B06m9s) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EZ_EOX4LESRMqM68Cx5u9pcBUuwHjokiQkMKadQD0sCzZA?e=oi7nHU) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ESEIGLXax7ZCowLqYLnepXMBuAdNpu40zOY5i3fI3ZHlaA?e=equtYL) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EaxV_UYnqqNCqW_nWQ88bSQBzrldAxL3QRykF1QMUlPIjQ?e=lzIlsc) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ES-w5btXXfFElHFCMRzx044Bz0oymKtZnHgVqg2rnerc4Q?e=YScTYg) |
| 🪲 | Conformer         | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EcgURZXQnW5PlM1a9kiPCrIBaOJ-n5LfMSw5cuHYU1ER6g?e=FAcuqb) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Ea881Ig2zYlHnl11hsfG-18B803STQDW3toSXKK391Po-w?e=VrCsgB) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EfH2ttIfIBFOlnXcQF6pBocBhJl3TwZd_MH3ZdZlpbedpQ?e=L2wEgV) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EaQHgqXzd4ZGhMiUXOksMEgB2D4RSEeuiRBOCyKL-bkHVg?e=cXtKkA) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EUWrB3K6dYtOkoh1oKrQKQIBMiwZxm5tQ18yJottinl6xA?e=Lsyd5K) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EWmaH-3kSNNDrLb2n1qD3JAB-yfzps5jbJsIi9kvE03KBA?e=wYCBi4) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ES9zRvqd8DVMm0zPRDxbix8BgqIXTP8CJKmgBVX2yH_EKw?e=HxWH26) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EYlavjz11ChJtUqRdGRYXskB6q1RcUU0___nbylFch7FCw?e=vGeRJN) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EfpXTXTQFaZFhMNceQx3hAQBtutgSY7ZVfEMTfgXi_QjTw?e=MyVVK2) |
| 🪲 | + CTC Compression | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EbCm-j8pwTVLjXuK5t576fgBeDXZVx5zg4S-_kLhV72hcg?e=ryIgJH) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EWqNkjhDk1pDhCD3wSc9Zj8BkIIXTCxk5Ch1QlsP7zzIBA?e=B065wp) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Ed5z6D1RzDxLlCcuG0XHCu0B3hcCJeYJnJzJRO9erQHW-w?e=yIcdo7) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EcuEuwn96X5Ft9PXXTSVedUBGgBZmLq3JaTrBYAG7cfPGQ?e=E9yiVL) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EeNtQkko-AhLqqGq08H7n1ABTx3Gnur4p7lmyBlJbd30fA?e=rVSC1z) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ERbFaYtgwEpFm252_uirhawBqmwJq-s9GmuTFiaaYy5fHw?e=Qmn05G) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EUrRl0AvNq5GnhxgJeWMKQkBIOAtz3TkfHFrC2_OE61hXA?e=pmochx) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Ee_d96OJ_JZHjvyVM4TdFRABr-6qDAQMgFulyStR7hU7yQ?e=zfC9jr) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EZInDa9LxU9Fq-t4Q55EgxkB2SnUhN435fWEffMptjp-_w?e=c3xmLo) |

## Citation
```bibtex
@article{papi2023reproducibility,
      title={{Reproducibility is Nothing without Correctness: The Importance of Testing Code in NLP}}, 
      author={Sara Papi and Marco Gaido and Andrea Pilzer and Matteo Negri},
      year={2023},
      url={https://arxiv.org/abs/2303.16166},
      journal={arXiv preprint arXiv:2303.16166},
}
```
