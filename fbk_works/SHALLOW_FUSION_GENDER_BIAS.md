# Shallow Fusion with Gender-Specific LM to Mitigate Gender Bias (EMNLP 2023)

Code and models for the paper:
[**Integrating Language Models into Direct Speech Translation: 
An Inference-Time Solution to Control Gender Inflection**](https://arxiv.org/abs/2310.15752v1)
accepted at EMNLP 2023.


## Models

To ensure reproducibility, we release the model checkpoints used in our experiments,
together with the SentencePiece model, the vocabulary files, and the yaml files:
- **Baseline ST Models**: [en-es](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EcDXRCaV4m9MqDZDUziWk7YB2cBwp2Px6NY_eFbRNj1tSA?e=oVqxzY),
[vocab_src_txt](https://fbk-my.sharepoint.com/:t:/g/personal/dfucci_fbk_eu/ETUbJ1Up0HxAlFHJeibnCDsBQa_jCmGrRoh-RYSJ14nvuQ?e=8siU4y), [spm_src_model](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EYkFmRy5d-9BvAYyoSjMrDoBufj4_wC-A00X5C3gAP13KQ?e=OchY8h),
[vocab_tgt_txt](https://fbk-my.sharepoint.com/:t:/g/personal/dfucci_fbk_eu/Ea2pHVkEWoFGqx93rj9TTw0B9LfXcgCTDjDBrcRZNfaZXg?e=qJbajC), [spm_tgt_model](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ETpA0Ynx_LBBuqlsGkAyXKsBIY8Is37-hSvvkB-cjlKKBA?e=bhdgEM) |
[en-fr](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EbNA-4pHCm5HkxJDabbgCIsB_6GUKny8ucT4W0EvrkQWzw?e=ZQLzcQ),
[vocab_src_txt](https://fbk-my.sharepoint.com/:t:/g/personal/dfucci_fbk_eu/EeOSsAx_KKpOgGQuofC04_8BIPUqC6gJSw4igBvnNrGtCw?e=PH8vwZ), [spm_src_model](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ESI_HTQ9oAVMrAy69v1XlNgBW9bmNXvK2pCuQNy2bmXXWA?e=T3acyA), 
[vocab_tgt_txt](https://fbk-my.sharepoint.com/:t:/g/personal/dfucci_fbk_eu/EYNb1GFe46tAlSP-DZsOGwgBkG1RzkdQjLJsrQiKOtyuRg?e=Y0STTC), [spm_tgt_model](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EXQUA2DeVNJNpIvoqARrSk4B_kBgF1QngWjCYT5S_5xhfQ?e=3rwbSB) |
[en-it](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EW8NLZO0xshPjQZtj1ZNNXUBOBeNgjcQ_bKbc1m837W53w?e=TeNkFt),
[vocab_src_txt](https://fbk-my.sharepoint.com/:t:/g/personal/dfucci_fbk_eu/EelnpJXTtrNMnOKEoIm475wBrb8kCz06rU-FtL8HW5dpLw?e=zYBLpK), [spm_src_model](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EbGGwMccLH9GgJfpQS3OMuwBMsmen3FpGwhTmfvFSpi3eQ?e=NZA1af), 
[vocab_tgt_txt](https://fbk-my.sharepoint.com/:t:/g/personal/dfucci_fbk_eu/EXlf-eZMgVFHop4hSzJPVasBAsNC-o4WXvZaZfsX55SUlw?e=FuthQK), [spm_tgt_model](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EVv1ftuBOSREs6BcP80iKLsBFIfKq6Or0h2x0Ujg0OQuCA?e=AsIr9L) |
[config_file](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ET3L78C9LCpHsp9rsvjVU3IBJk8dFiXmkMnvWZOKf6w-_A?e=GZUYQl)
- **Specialized ST Models**: en-es: [M](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EQasJ8PW0nlNtgtIHPlCfJ0Bgqgmv0VX8dCn4ZS5Ox_Uxw?e=tOAuDo), [F](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EV3kY0-ufv9Jk7v1WGPkE1cB-8tBCyEZzF1Ruj2yjWJvYg?e=DYGbxQ) | 
en-fr: [M](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EdjvXBG1h8NAgRflBWQLhCgBMjo-jKEL1ilhBqsvlvMqsg?e=3GaWgO), [F](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EUiQR0momOdBg2xsNM1k3OYBKXx32pTdzqJKYxu6JhO9kw?e=u1t10A) | 
en-it: [M](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EQksjXQCRilCsY9GsqdDk10BH2YAKTZb7wlK5Hh5YaeV2A?e=3GKsu2), [F](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ESYdZ7hFAk9Bj4zF8kw0i7YB7UOk7ias8VsNyDu-41nxrQ?e=C5h59i)
- **Language Models**: es: [M](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ERZAXNUR8XNCsCd6TK2sTCgBSMmuyOWQDcJgjM_eVGZecQ?e=PrwhKz), [F](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EWWLefKIv4ZNo6ZhjnwVqc4B3eDt4dFVkCL6vty4udwjiA?e=y7YWZ7) | 
fr: [M](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ERjieMoB4z9HowQIxmQ8e3wBfqsrGmZtZ67WcIc0jADaPg?e=YcJg86), [F](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EWNkjJvWtTRDr2fErUPQ8XIBFrf4751gLO4XhRbr4tXg8w?e=O309kq) | 
it: [M](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EXM6-sPQJ-pEiHT1PbR-la4BmcjAm3qqeyDMpKkFuMYhHg?e=GqH1Tm), [F](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EbxrARzsnO9Lvk7Yr6r-RdIBoIAeFXrZNJbGCESmX0QxVA?e=OeNqFv)


## How to run

With the following instructions, it is possible to reimplement our models from scratch,
as well as to use our models for inference.


### Data and Preprocessing

#### ST Data

To train the ST models, we have used the [MuST-C dataset](https://mt.fbk.eu/must-c/). \
The data, comprising text files found in `$MUSTC_TEXT_DATA_FOLDER` and audio files in `$MUSTC_WAV_FOLDER`, 
can be preprocessed for each target language (`$TGT_LANG`) using the following command. 
The results will be saved in `$MUSTC_SAVE_FOLDER`.

```bash
python ${FBK_FAIRSEQ}/examples/speech_to_text/preprocess_generic.py \
      --data-root $MUSTC_TEXT_DATA_FOLDER \
      --save-dir $MUSTC_SAVE_FOLDER \
      --wav-dir $MUSTC_WAV_FOLDER \
      --split train dev \
      --vocab-type bpe \
      --vocab-size 8000 \
      --src-lang en --tgt-lang $TGT_LANG \
      --task st \
      --n-mel-bins 80
```
 
For testing the ST models, we have used the [MuST-SHE](https://mt.fbk.eu/must-she/) test set. \
The text files located in `$MUSTSHE_EXT_DATA_FOLDER` and the audio files in `$MUSTSHE_WAV_FOLDER`
which can be preprocessed for each target language (`$TGT_LANG`) using the text vocabularies (`$VOCAB_SRC` and 
`$VOCAB_TGT`) obtained by the previous preprocessing step. 
The results will be saved in `$MUSTSHE_SAVE_FOLDER`.

```bash
python ${FBK_FAIRSEQ}/examples/speech_to_text/preprocess_generic.py \
      --data-root $MUSTSHE_EXT_DATA_FOLDER \
      --save-dir $MUSTSHE_SAVE_FOLDER \
      --wav-dir $MUSTSHE_WAV_FOLDER \
      --split MONOLINGUAL.${TGT_LANG}_v1.2 \
      --vocab-type bpe \
      --vocab-file-src /$VOCAB_SRC \
      --vocab-file-tgt $VOCAB_TGT \
      --src-lang en --tgt-lang $TGT_LANG \
      --task st \
      --n-mel-bins 80
```

#### Monolingual Text (LM) Data

To train the LMs we have used [GenderCrawl](https://mt.fbk.eu/gendercrawl/), a set of text corpora 
derived from [ParaCrawl](https://www.paracrawl.eu/) by selecting the sentences that contain 
gender-marked words referring to the speaker. \
These data can be preprocessed using the following command, where `$TRAIN_DATA_TOKENIZED` and
`$DEV_DATA_TOKENIZED` are the text training and validation data tokenized with the SentencePiece model
obtained by the ST preprocessing, `$VOCAB_SRC` is the txt vocabulary obtained used for the ST preprocessing, 
and `$GENDERCRAWL_SAVE_FOLDER` is the folder where the preprocessed data are stored.

```bash
fairseq-preprocess \
        --task language_modeling \
        --cpu \
        --only-source \
        --trainpref $TRAIN_DATA_TOKENIZED \
        --validpref $DEV_DATA_TOKENIZED \
        --srcdict $VOCAB_SRC \
        --destdir $GENDERCRAWL_SAVE_FOLDER
```

### Training

#### Base ST Models

To train the base ST models we have used the following command 
(parameters intended for training on 4 GPUs, each with 40 GB of VRAM).
The `$TRAIN_MUSTC` and `$DEV_MUSTC` files are in TSV format located in `$MUSTC_FOLDER`, 
obtained after preprocessing. The `$CONFIG_ST` is a YAML file and can be downloaded above.
Final checkpoint and log information will be saved in `$ST_BASE_SAVE_DIR`.

```bash
python ${FBK_FAIRSEQ}/train.py $MUSTC_FOLDER \
        --train-subset $TRAIN_MUSTC \
        --valid-subset $DEV_MUSTC \
        --save-dir $ST_BASE_SAVE_DIR \
        --num-workers 3 \
        --max-update 50000 \
        --max-tokens 40000 --adam-betas '(0.9, 0.98)' \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc \
        --config-yaml $CONFIG_ST \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch conformer \
        --ctc-encoder-layer 8 --ctc-weight 0.5 --ctc-compress-strategy avg \
        --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
        --warmup-updates 25000 \
        --clip-norm 10.0 \
        --update-freq 2 \
        --skip-invalid-size-inputs-valid-test \
        --log-format simple >> $ST_BASE_SAVE_DIR/train.log 2> $ST_BASE_SAVE_DIR/train.err

python ${FBK_FAIRSEQ}/scripts/average_checkpoints.py --input $ST_BASE_SAVE_DIR \
        --num-update-checkpoints 7 \
        --checkpoint-upper-bound 50000 \
        --output $ST_BASE_SAVE_DIR/avg7.pt
```

#### Specialized ST Models

To finetune the base ST models and develop the specialized ST models, we used the following
command, where `$TRAIN_MUSTC_${GDR}` and `$DEV_MUSTC_${GDR}` are the gender-specific portions of MuST-C
TSV files obtained with the [MuST-Speaker](https://mt.fbk.eu/must-speakers/) resource.
Final checkpoint and log information will be saved in `$ST_SPECIALIZED_SAVE_DIR`.

```bash
python ${FBK_FAIRSEQ}/train.py $MUSTC_FOLDER \
        --train-subset $TRAIN_MUSTC_${GDR} \
        --valid-subset $DEV_MUSTC_${GDR} \
        --save-dir $ST_SPECIALIZED_SAVE_DIR \
        --num-workers 3 \
        --max-epoch 7 \
        --max-tokens 40000 --adam-betas '(0.9, 0.98)' \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc \
        --finetune-from-model $ST_BASE_SAVE_DIR/avg7.pt \
        --config-yaml $CONFIG_ST \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch conformer \
        --ctc-encoder-layer 8 --ctc-weight 0.5 --ctc-compress-strategy avg \
        --optimizer adam --lr 1e-3 \
        --warmup-updates 25000 \
        --clip-norm 10.0 \
        --update-freq 2 \
        --skip-invalid-size-inputs-valid-test \
        --log-format simple >> $ST_SPECIALIZED_SAVE_DIR/finetuning.log 2> $ST_SPECIALIZED_SAVE_DIR/finetuning.err

python ${FBK_FAIRSEQ}/scripts/average_checkpoints.py --input $ST_SPECIALIZED_SAVE_DIR \
        --num-epoch-checkpoints 4 \
        --checkpoint-upper-bound 7 \
        --output $ST_SPECIALIZED_SAVE_DIR/avg4.pt
```

#### Language Models

To train the LMs, we have used the following command (parameters intended for training
on 1 GPU with 12G of VRAM).
`$TRAIN_GENDERCRAWL` and `$DEV_GENDERCRAWL` are the bin files obtained from the preprocessing and are located in
`$EGOCRAWL_FOLDER`. The final checkpoint will be saved in $LM_SAVE_FOLDER.

```bash
fairseq-train $GENDERCRAWL_FOLDER \
        --task language_modeling \
        --train-subset $TRAIN_GENDERCRAWL \
        --valid-subset $DEV_GENDERCRAWL \
        --validate-interval 10000 --validate-interval-updates 100 \
        --save-dir $LM_SAVE_FOLDER \
        --arch transformer_lm --share-decoder-input-output-embed \
        --dropout 0.1 \
        --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay 0.01 --clip-norm 0.0 \
        --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 200 --warmup-init-lr 1e-07 \
        --tokens-per-sample 1024 --sample-break-mode none \
        --max-tokens 16384 --update-freq 8 \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --patience 5 \
        --fp16 \
        --save-interval-updates 100 \
        --keep-interval-updates 10 \
        --no-epoch-checkpoints
        
 python ${FBK_FAIRSEQ}/scripts/average_checkpoints.py --input $LM_SAVE_FOLDER \
        --num-epoch-checkpoints 5 \
        --checkpoint-upper-bound $(ls $LM_SAVE_FOLDER | head -n 5 | tail -n 1 | grep -o "[0-9]*") \
        --output $LM_SAVE_FOLDER/avg5.pt       
```


### Inference

For the **base** and **specialized ST models**, whose checkpoint is `$CHECKPOINT`, we have used the following
command, where `MUSTSHE_DATA` is the preprocessed TSV file located in
`$MUSTSHE_SAVE_FOLDER`, `CONFIG_FILE` is the YAML file provided above. 
The output translations will be saved in `$OUTPUT`.

```bash
python ${FBK_FAIRSEQ}/fairseq_cli/generate.py $MUSTSHE_SAVE_FOLDER \
        --gen-subset $MUSTSHE_DATA \
        --user-dir examples/speech_to_text \
        --max-tokens 20000 \
        --config-yaml $CONFIG_FILE \
        --beam 5 \
        --max-source-positions 10000 \
        --max-target-positions 1000 \
        --task speech_to_text_ctc \
        --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --no-repeat-ngram-size 5  \
        --path $CHECKPOINT > $OUTPUT
```

For shallow fusion, inference can be executed with the following command. Here,
`$AVG_ENCODER` is the averaged encoder output provided above, and `$ILMW` and `$ELMW` are the weights for the 
internal LM contribution and the external LM contribution, respectively.
The output translations will be saved in `$OUTPUT_SHALLOW_FUSION`.

```bash
python ${FBK_FAIRSEQ}/fairseq_cli/generate.py $MUSTSHE_SAVE_FOLDER \
        --gen-subset $TEST_DATA \
        --user-dir examples/speech_to_text \
        --max-tokens 20000 \
        --config-yaml $CONFIG_FILE \
        --beam 5 \
        --max-source-positions 10000 \
        --max-target-positions 1000 \
        --task speech_to_text_ctc_noilm \
        --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --no-repeat-ngram-size 5  \
        --path $ST_BASE_SAVE_DIR/avg7.pt \
        --lm-path $LM_SAVE_FOLDER/best.pt \
        --ilm-weight $ILMW \
        --lm-weight $ELMW \
        --encoder-avg-outs $AVG_ENCODER > $OUTPUT_SHALLOW_FUSION
```

### Evaluation

We have used [SacreBLEU](https://github.com/mjpost/sacrebleu) 2.0.0 to compute BLEU
and the official script of MuST-SHE to compute _gender accuracy_ and _term coverage_ metrics. \
To run the paired bootstrap resampling we have used the implementation in SacreBLEU for BLEU scores,
and the following command for the gender accuracy and term coverage scores (`$MUSTSHE_REF` is the text file
containing reference sentences, `$BASE_OUTPUTS` and `$EXP_OUTPUTS` are the text files containing
the translated sentences by the two systems to be compared, and `$CATEGORIES` is the categories to be evaluated).

```bash
python ${FBK_FAIRSEQ}/examples/speech_to_text/scripts/gender/paired_bootstrap_resampling_mustshe.py \
        --reference-file $MUSTSHE_REF \
        --baseline-file $BASE_OUTPUTS \
        --experimental-file $EXP_OUTPUTS \
        --categories $CATEGORIES \
        --num-samples 1000 \
        --significance-level 0.05
```


## Citation

```bibtex
@inproceedings{fucci-etal-2023-integrating,
    title = "Integrating Language Models into Direct Speech Translation: An Inference-Time Solution to Control Gender Inflection",
    author = "Fucci, Dennis and 
        Gaido, Marco and 
        Papi, Sara and 
	Cettolo, Mauro and
	Negri, Matteo and 
        Bentivogli, Luisa},
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
```
