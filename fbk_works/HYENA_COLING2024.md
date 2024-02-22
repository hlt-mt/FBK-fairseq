# How do Hyenas deal with Human Speech? Speech Recognition and Translation with ConfHyena

This README contains the instructions to replicate the training and evaluation of the models in the paper
[How do Hyenas deal with Human Speech? Speech Recognition and Translation with ConfHyena](https://arxiv.org/abs/2402.13208).
In addition, we release the pre-trained models used in the paper.


## Setup
Clone this repository and install it as explained in the original [Fairseq(-py)](https://github.com/pytorch/fairseq).
For the experiments we used MuST-C, make sure to [download the corpus](https://mt.fbk.eu/must-c/).
Follow the [preprocessing steps of Speechformer](SPEECHFORMER.md#preprocessing) to preprocess the MuST-C data.

## Pretrained models

Below we release the dictionary/config files and the pre-trained checkpoints
obtained in our experiments.
The dictionary and config files are the same as those used for the Conformer baseline,
whose checkpoints can be found [here](BUGFREE_CONFORMER.md#pretrained-models).

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
| Model              | en (ASR)   | en-de      | en-es      | en-fr | en-it | en-nl | en-pt | en-ro |
|--------------------|------------|------------|------------|-------|-------|-------|-------|-------|
| ConfHyena          | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EU6Bhy_jGQxJm9fIS3DsJmwBxd-tBl5HsQBM2OCbvu5gQQ?e=cORIdz) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ETuTNLx7_hNAooQ_U5yQh1oB3zae2fls2xv-K4enmCBMRw?e=2ENGAV) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXWYMvNOEINMgeKStlW0peABybfiOIcOpInjpbFw3cRUBw?e=JyPdry) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Eb3pv7C6zvJIqkH2nPa9w4YBvvO74khSX7s_uo6D_p7fzg?e=t7NypZ) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Ee7Gvuo2iRJHsr2M_9G4KHQBkgrRkCmwCy5kS9jMlJVP6A?e=lbVxTr) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ERpXg6Cbe3pDlL1gzGCoe7UBHcpCLw2JQXQKtK1vF05NGg?e=RSsHDJ) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ETXG5TySDzZLmaWkaPbFMXgBIoxE3n54I-pclaRsmQQedg?e=JNdKaE) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EdR34a0DMMhGiRsIpGCxcQABIbjbICogJaTKZXOtGQa14w?e=MYRU3N) |
| - non-causal Hyena | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Efpj8KHH9oJDm6bPAJSdDNkB_JRcsmcxXC4ciaPE0U3kgg?e=yfGbhq) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EYHfCm2e4PBAoE-0jHkEm2MB1Wr-qBZAEaeAWJBUXl30Lg?e=ZuKaon) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EVlb_DmkG8VCg2JrddtHGOoB9be1IDpB2Q0aQavIe6hoAw?e=aL3SWY) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EawfiABptKBErrLwYJ5fjdUBAVSVv1gsWU-jwWlgj8qt_A?e=hnU9HB) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EQZSTWx_O8RFhFICdyE8swkBsrCmwkA0LouzRnX4cF7wHQ?e=0Ha4zB) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EXTo4vC_hMtAgijZ1TE7RWABZgwfI4wuXrZvlcHI_ah7Lg?e=3Baczg) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EUf-qg9uf_VBgAZDS5OW3DIBRK8gkxts-Ku067r00bb1VQ?e=2uNgZj) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Ea8i1u8KIldLreB5531Fno8BCEpg7qiiHG2lCE8cE8qZXA?e=pTXNCC) |
| Hybrid ConfHyena   | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EbF2LjOz1MtLnX1gCHTQjsEBgLn_EAhKypyIDhu3Y7nuFQ?e=ZhFRyF) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ETmOsR9Ie6hOrhM50B6wzioBvWuSLo6g55e_qIp88W13qQ?e=W5eK79) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EcYJBCcrJaNApvOlvWDnRSEBtue-fzMIYpISwMWqdRCPSQ?e=gWXrK5) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EfSjqbL1CwZOkquHsvwZnpQBswt469ymSW3uL_q8ro5xlg?e=GMHKPO) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EQXC8NAnPldElP_0WduGtGYB2lhKCCy-tOQQDBfeQMvC4A?e=0T63hZ) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EZin2xLqqLFBkFYPyh0X1rcBCFbdvB-Dpr567adjGkrpSQ?e=57imQ7) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EVjU7IhkWB5Dq7M09SzWpqABn18U_GbSGdj4biJoNWCaJw?e=vQpsEh) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EYROvhAPNTxEn9WDHgpIgPEBsKhWUWYTpEfydwFV9AXDIw?e=oaet0d) |
| - non-causal Hyena | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EbNqXhyaUGVFheZ3FExAloEBPEZOG2jlpJv8ynnYnYpf2g?e=qe87Zq) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Ef5HXS1LJvxNvYHv-bp-cNUBZ4DDGdWBAL_iBQNpl6JbcA?e=DX0ItZ) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EWQ_V6szbMdPp149zGa8tuoBLnN-nZ0tVnYc3ymBb9Ddcg?e=BByutz) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EVsXeLu_VkFKndxmUAShl1kB7ANPmdw19QOA87RUBP-TcQ?e=q8Royw) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EbTZHMRnb_BJobUxSK0dFScB3FD1_IvVcLvyfnIWFy6lPg?e=mqh2wK) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/Ef_gZRguYWJEmzyIMn9bzIUBzGgCt-lwb_5FPCSrUHv03A?e=LssC98) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EVhOSMYNNqlJibkYt85laRoBvwNrNzvXCAOX_CJYX13_MQ?e=9KfTZJ) | [ckp.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EZ64nTKeOvhBmNzZgyx7LV8BJKJN0Qx0psoqLYaJ7lzPlg?e=Yq8tAv) |



## Training

For the Conformer baseline, please refer to the [bug-free Conformer README](BUGFREE_CONFORMER.md).

For the Hybrid ConfHyena models, our training has been executed with the following commands.


```bash
LANG=$1
MUSTC_ROOT=$2
TASK=$3
SAVE_DIR=$4

mkdir -p $SAVE_DIR

python ${FBK_fairseq}/train.py ${MUSTC_ROOT} \
        --train-subset train_${TASK}_src --valid-subset dev_${TASK}_src \
        --user-dir examples/speech_to_text --seed 1 \
        --num-workers 2 --max-update 100000 --patience 10 --keep-last-epochs 12 \
        --max-tokens 40000 --update-freq 4 \
        --task speech_to_text_ctc --config-yaml config.yaml  \
        --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch confhyena --conformer-after-compression --stride 2 \
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

The ConfHyena models can be obtained by removing the `--conformer-after-compression` parameter.


The causal version of the two architectures (`- non causal Hyena` in the paper and tables below)
can be obtained by adding the parameter `--hyena-causal` to the command.

The command is meant to be executed on 2 A100 GPUs with 40GB VRAM.


## Evaluation
Once you downloaded the pretrained checkpoints and related config/dictionaries,
generate the output with:
```bash
python ${FBK_fairseq}/fairseq_cli/generate.py ${MUSTC_ROOT} \
        --user-dir examples/speech_to_text \
        --config-yaml config.yaml --gen-subset tst-COMMON_st_src \
        --max-source-positions 10000 --max-target-positions 1000 \
        --task speech_to_text_ctc \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
        --beam 5 --no-repeat-ngram-size 5 --path ${PRETRAINED_CHECKPOINT} > ${OUTPUT_FILE}
```

## Citation
```bibtex
@inproceedings{gaido-et-al-2024-hyena,
      title={{How do Hyenas deal with Human Speech? Speech Recognition and Translation with ConfHyena}}, 
      author={Marco Gaido and Sara Papi and Matteo Negri and Luisa Bentivogli},
      year={2024},
      address="Turin, Italy",
      booktitle={Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)},
}
```
