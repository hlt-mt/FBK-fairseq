# SBAAM! Eliminating Transcript Dependency in Automatic Subtitling (ACL 2024)

Code and models for the paper "[SBAAM! Eliminating Transcript Dependency in Automatic Subtitling](https://arxiv.org/abs/2405.10741)"
published at ACL 2024.

## 📌 Pretrained models 
### 📎 MuST-Cinema (multilingual)
- **English > {Dutch, French, German, Italian, Portuguese, Romanian, Spanish}**: [model.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EeO1bNCnhjtMuIPaE6ApTOIBHmPrD42pGtCLIRewBWTTyg?e=u0k8rs) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EZHmdY4GH-FHrUfKoKzy8AAB0iORJQ3nLCxv6P0g_1xUWA?e=0eEvYY) | [src_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/ERs15CM9DDJHnrgtm1aTNggBNIVUwxky_RShW-BygBSlSQ?e=fM9vEi) | [src_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EbFH9eQvkz5CicnzWTZL7aIB8n1VWCzlmy5RHj6KKuIJUw?e=15JKFC) | [tgt_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/Ef-ZRJ2QOY1Ev9fgtA6QdeYB10cGkvQJdJ2kq0A9Nky-QA?e=5WqVd8) | [tgt_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EWc52NhLTjpDnSdOPDDJZlMBwA0ZFcEkc9a3glhM4FNNig?e=u52Ih8)
### 📎 Unconstrained (or Constrained setting of IWSLT 2023)
- **English > German**: [model.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EVOFOmltFMNNu0AK_uBs0o8B0tE5O02uuVSuz8k7qdhZow?e=nCsHqB) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EfaxwABEFXFHq3nN-K_AlasBDcLocYCiqybDLJUZFCEq6A?e=qBomMR) | [src_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EQ8c_ndPrzZLhlDrC301FxABZ2WI7Mx8-JWAw02a8m2KSw?e=F5wcjf) | [src_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EWcBzc8gPr9PqVd-0ok6RLAB8hxjUIm39x-xZcABI85cMQ?e=4dJ511) | [tgt_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EfwsV1_m3EJMu2vD4z7dE90B6FjsDZ230lip0MYQ2VFR7A?e=stctUG) | [tgt_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EfHR-AXf9b9Hr3gGRHCDyIsBoMYW8A6R3kHcP0Yl60o_hA?e=pj4wna)
- **English > Spanish**: [model.pt](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EYHffzXEtstCjqNBNMTgND4B6aOqMo5nGrU5WH9EQ2zGwA?e=pUyKXv) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EfjKms5f07tNm1XWdjeNVMYB94xILSES2L1ZHnJzVpv_hQ?e=CnOhju) | [src_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EQ8c_ndPrzZLhlDrC301FxABZ2WI7Mx8-JWAw02a8m2KSw?e=F5wcjf) | [src_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EWcBzc8gPr9PqVd-0ok6RLAB8hxjUIm39x-xZcABI85cMQ?e=4dJ511) | [tgt_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EeKpDfqyW7ZIglfsVUuFk1IBj2vYd2W_VYlIwasOP3EZgg?e=DCoIcp) | [tgt_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/mgaido_fbk_eu/EbLaVF-w8KhAlCFyiGqzUkMB-CCG6_8-Ep4MJplW4hDg0g?e=NHglAy)

## 📍 Setup and Data Preprocessing
1. Clone this repository and install it as explained in the original [Fairseq(-py)](https://github.com/pytorch/fairseq).
2. Download all the corpora listed in our paper and preprocess them as explained [here](SPEECHFORMER.md#preprocessing). 

## 🏃 Training

### Multilingual Model on MuST-Cinema

For the training of the multilingual model on MuST-Cinema, we fine-tuned the multilingual model described
in [Direct Speech Translation for Automatic Subtitling](DIRECT_SUBTITLING.md).
To replicate it from scratch, please follow that readme to build the base model and then fine-tune the model
with the following script:

```bash
python ${FBK_fairseq}/train.py ${DATA_ROOT} \
        --train-subset train_multilang_de,train_multilang_es,train_multilang_it,train_multilang_fr,train_multilang_pt,train_multilang_nl,train_multilang_ro \
        --valid-subset dev_multilang_de,dev_multilang_es,dev_multilang_it,dev_multilang_fr,dev_multilang_pt,dev_multilang_nl,dev_multilang_ro \
        --save-dir ${ST_SAVE_DIR} \
        --num-workers 2 --max-update 200000 \
        --save-interval-updates 1000 \
        --max-tokens 20000 --adam-betas '(0.9, 0.98)' \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc --config-yaml ${CONFIG_YAML} --ignore-prefix-size 1 \
        --criterion ctc_multi_loss --underlying-criterion joint_cross_entropy_ctc \
        --label-smoothing 0.1 --best-checkpoint-metric loss \
        --arch conformer_joint_ctc \
        --ctc-encoder-layer 8 --ctc-weight 1.0 --primary-loss-weight 5.0 --auxiliary-loss-weight 2.0 \
        --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
        --warmup-updates 25000 --patience 20 --keep-interval-updates 25 --no-epoch-checkpoints \
        --clip-norm 10.0 \
        --seed 1 --update-freq 8 --allow-partial-loading --allow-extra-tokens \
        --skip-invalid-size-inputs-valid-test --pretrained-model ${DIRECT_BASE_MODEL}  \
        --log-format simple >> $ST_SAVE_DIR/train.log 2>$ST_SAVE_DIR/train.err
```

where `${FBK_fairseq}` is the folder containing this repository,
`${DATA_ROOT}` is the folder containing the preprocessed datasets, 
`${ST_SAVE_DIR}` is the directory in which the checkpoints will be saved,
`${CONFIG_YAML}` is the path to the yaml file generated after preprocessing,
and `${DIRECT_BASE_MODEL}` is the path to the pre-trained model obtained after following the other readme.
Note that the names of the TSV files specified in `--train-subset` and `--valid-subset` may be different in your case.

This script is intended for 2 NVIDIA A100 40GB, please set `--max-tokens` and `--update-freq` accordingly with 
your hardware, so that `number of GPUs * max_tokens * update_freq = 320,000`.

### Bilingual Models in IWSLT Constrained Conditions

The overall process to replicate our models from scratch includes 3 training phases:

1. An ASR training
2. An ST training (with the encoder weights initialized from the ASR)
3. Subtitling fine-tuning from the ST model with the inclusion of the CTC on target module

Below, we report the scripts we used for each of the three phases.
The scripts include the name of the TSV of the datasets we used. Replace them accordingly
with the names of the TSVs that you obtain after the preprocessing.

All the scripts have been run on 4 A100 GPUs with 64GB VRAM. As mentioned above,
set `--max-tokens` and `--update-freq` accordingly with your hardware,
so that `number of GPUs * max_tokens * update_freq = 320,000`.

#### ASR Training

The ASR training script is listed below. At the end of the training, we average the last 7 checkpoints
with the [dedicated script](../scripts/average_checkpoints.py).

```bash
train_tsv=train_commonvoice,train_covost2,train_europarl,train_librispeech,train_mustcinema,train_mustcv2,train_tedlium,train_voxpopuli
dev_tsv=dev_mustcv2

python ${FBK_fairseq}/train.py ${DATA_ROOT} \
	--train-subset $train_tsv --valid-subset $dev_tsv --config-yaml $config \
	--save-dir $save_dir --user-dir ${FBK_fairseq}/examples/speech_to_text \
	--task speech_to_text_ctc --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 --ctc-encoder-layer 8 --ctc-weight 0.5  \
	--arch conformer \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 --adam-betas '(0.9, 0.98)' --weight-decay 0.001 \
	--seed 1 --skip-invalid-size-inputs-valid-test \
	--update-freq 2 --max-tokens 40000 --num-workers 4 \
	--max-update 250000 \
	--keep-interval-updates 25 --no-epoch-checkpoints --save-interval-updates 1000 \
	--log-format simple >> $ASR_SAVE_DIR/train.log 2> $ASR_SAVE_DIR/train.err
```

The checkpoint we obtained at the end of this step can be downloaded [here](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EYUt4Twv4d1DiEE1scdrBtEBn8IliLDuXHZfLOorHBBeiw?e=LQXyU8).

#### ST Training

The ST training script is similar to the ASR one, but here we load the encoder weights
of the ASR obtained from the previous step. Also in this case, at the end of the training,
we average the last 7 checkpoints with the
[dedicated script](../scripts/average_checkpoints.py).

All the datasets with the `_nemo` suffix indicate that the target translation
has been obtained by translating the ASR transcripts with the 
[NeMo MT models](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/nlp/machine_translation/machine_translation.html).

```bash
if [ "$lang" == "de" ]; then
  dev_tsv=dev_mustcv2
  train_tsv=train_commonvoice_nemo,train_covost2,train_europarl,train_librispeech_nemo,train_mustcinema,train_mustcv2,train_tedlium_nemo,train_voxpopuli_nemo
else
  dev_tsv=dev_mustc
  train_tsv=train_commonvoice_nemo,train_covost2_nemo,train_europarl,train_librispeech_nemo,train_mustcinema,train_mustcv2_nemo,train_tedlium_nemo,train_voxpopuli_nemo
fi

python ${FBK_fairseq}/train.py ${DATA_ROOT} \
	--train-subset $train_tsv --valid-subset $dev_tsv --config-yaml ${CONFIG_YAML} \
	--save-dir $save_dir --user-dir ${FBK_fairseq}/examples/speech_to_text \
	--task speech_to_text_ctc --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 --ctc-encoder-layer 8 --ctc-weight 0.5  --ctc-compress-strategy avg \
	--arch conformer \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 --adam-betas '(0.9, 0.98)' --weight-decay 0.001 \
	--seed 1 --skip-invalid-size-inputs-valid-test \
	--update-freq 2 --max-tokens 40000 --num-workers 4 \
	--max-update 200000 \
	--load-pretrained-encoder-from ${ASR_MODEL_PATH} \
	--keep-interval-updates 25 --no-epoch-checkpoints --validate-interval 1000 --save-interval-updates 1000 \
	--log-format simple >> $ST_SAVE_DIR/train.log 2> $ST_SAVE_DIR/train.err
```

The checkpoints that we obtained after this step are
[here for German](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/ESwZg4vYMMBAlmOlV5kKKaABW-uLRiyX0KrPUEVLwStHrg?e=Nfvy0x)
and [here for Spanish](https://fbk-my.sharepoint.com/:u:/g/personal/mgaido_fbk_eu/EV_3itJkbQxDuTqza2yvEiUBqk4LEVdEgbqdTt-gYp1zhw?e=f6tGTf).

#### Subtitling Fine-tuning

Lastly, we fine-tune the ST model on the subtitle data (which includes the `<eob>` and `<eol>` tokens),
also adding the CTC on target module, which is randomly initialized. MuST-Cinema already contains the subtitle segmentation markers, 
while all the other datasets have to be segmented into subtitles 
[using the multimodal segmenter](SUBTITLE_SEGMENTER_AACL2022.md).

As in the previous steps, we average the last 7 checkpoints
with the [dedicated script](../scripts/average_checkpoints.py) after the training.

```bash
if [ "$lang" == "de" ]; then
  train_tsv=train_commonvoice_nemo_sub,train_covost2_sub,train_europarl_sub,train_librispeech_nemo_sub,train_mustcinema_sub,train_mustcv2_sub,train_tedlium_nemo_sub,train_voxpopuli_nemo_sub
else
  train_tsv=train_commonvoice_nemo_sub,train_covost2_nemo_sub,train_europarl_sub,train_librispeech_nemo_sub,train_mustcinema_sub,train_mustcv2_nemo_sub,train_tedlium_nemo_sub,train_voxpopuli_nemo_sub
fi
dev_tsv=dev_mustcinema_sub

python ${FBK_fairseq}/train.py ${DATA_ROOT} \
	--train-subset $train_tsv --valid-subset $dev_tsv --config-yaml ${CONFIG_YAML} \
	--save-dir $save_dir --user-dir ${FBK_fairseq}/examples/speech_to_text \
	--task speech_to_text_ctc --criterion ctc_multi_loss --underlying-criterion joint_cross_entropy_ctc \
	--label-smoothing 0.1 --ctc-encoder-layer 8 --ctc-weight 1.0 --primary-loss-weight 5.0 --auxiliary-loss-weight 2.0 --ctc-compress-strategy avg \
	--arch conformer_joint_ctc \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt --warmup-updates 25000 \
	--clip-norm 10.0 --adam-betas '(0.9, 0.98)' --weight-decay 0.001 \
	--seed 1 --skip-invalid-size-inputs-valid-test \
	--update-freq 2 --max-tokens 40000 --num-workers 4 \
	--max-update 200000 --allow-partial-loading --allow-extra-tokens \
	--pretrained-model `${ST_MODEL_PATH}` \
	--keep-interval-updates 25 --no-epoch-checkpoints --save-interval-updates 1000 --validate-interval 1000 \
	--log-format simple >> $SUBST_SAVE_DIR/train.log 2> $SUBST_SAVE_DIR/train.err
```

The checkpoints are available [at the beginning of this README](#-pretrained-models).

## 📺 Generation

First of all, segment the audio files of the test set with [SHAS](https://github.com/mt-upc/SHAS)
and [preprocess them](SPEECHFORMER.md#preprocessing).

Then, the generation of the SRT files is done in 3 steps:

1. We generate the subtitle texts (namely, the translation withe `<eob>` and `<eol>` delimiters);
2. We assign the start and end timestamp to the generated subtitle blocks;
3. We build SRT files using the two previous outputs.


### 1. Subtitle Texts Generation

In our paper, we experiment both with and without the rescoring with the joint CTC module.

To use the CTC rescoring, the command is:


```bash
python ${FBK_fairseq}/fairseq_cli/generate.py ${DATA_ROOT} \
	--user-dir ${FBK_fairseq}/examples/speech_to_text/ --config-yaml ${CONFIG_YAML} \
	--gen-subset $SPLIT --extract-attn-from-layer 3 --print-alignment soft \
	--max-tokens 100000 --unkpen 10000 --beam 5 \
	--model-overrides "{'batch_unsafe_relative_shift': False, 'pretrained_model': None}" \
	--max-source-positions 20000 --max-target-positions 1000 --avoid-repeated-eob-eol \
	--task speech_to_text_joint_decoding --ctc-decode-weight 0.2 --criterion ctc_multi_loss \
	--underlying-criterion joint_cross_entropy_ctc --no-repeat-ngram-size 5 \
	--path ${MODEL} > ${OUTDIR}/translation.out

grep "^D-" $OUTDIR/translation.out | cut -c3- | sort -k 1n | cut -f3 > $OUTDIR/translation.txt
```

While, to avoid the CTC rescoring, the command is:

```bash
python ${FBK_fairseq}/fairseq_cli/generate.py ${DATA_ROOT} \
	--user-dir ${FBK_fairseq}/examples/speech_to_text/ --config-yaml ${CONFIG_YAML} \
	--gen-subset $SPLIT --extract-attn-from-layer 3 --print-alignment soft \
	--max-tokens 100000 --unkpen 10000 --beam 5 \
	--model-overrides "{'batch_unsafe_relative_shift': False, 'pretrained_model': None}" \
	--max-source-positions 20000 --max-target-positions 1000 --avoid-repeated-eob-eol \
	--task speech_to_text_ctc --criterion ctc_multi_loss \
	--underlying-criterion joint_cross_entropy_ctc --no-repeat-ngram-size 5 \
	--path ${MODEL} > ${OUTDIR}/translation.out

grep "^D-" $OUTDIR/translation.out | cut -c3- | sort -k 1n | cut -f3 > $OUTDIR/translation.txt
```

In the case of multilingual models, add `--prefix-size 1` to the command. The `${CONFIG_YAML}` has to
be the one used for the training of the `${MODEL}`.

### 2. Timestamp Estimation

We tested three methods in our paper: SubCTC, attention-based DTW, and SBAAM.
Please refer to the paper for a detailed explanation of each of them.

To use SubCTC, run:

```bash
python ${FBK_fairseq}/examples/speech_to_text/scripts/ctc_align.py $DATA_ROOT \
        --user-dir examples/speech_to_text/ --config-yaml ${CONFIG_YAML} \
        --gen-subset $SPLIT \
        --max-tokens 10000 --beam 5 \
        --model-overrides "{'batch_unsafe_relative_shift': False, 'pretrained_model': None}" \
        --max-source-positions 10000 --max-target-positions 1000 \
        --split-tokens "<eob>" --feature-duration 0.04 \
        --task speech_to_text_ctc \
        --criterion ctc_multi_loss --underlying-criterion joint_cross_entropy_ctc \
        --use-target-text --ctc-method auxiliary \
        --path $MODEL --text-file ${OUTDIR}/translation.txt > ${OUTDIR}/translation_align.out

grep "^SEGM-" ${OUTDIR}/translation_align.out | cut -c6- | sort -k 1n | cut -f2 > ${OUTDIR}/translation_align.txt
```

While for the other two methods, use the following command:

```bash
python ${FBK_fairseq}/examples/speech_to_text/scripts/attention_based_timestamp_estimation.py \
        --fairseq-output $OUTDIR/translation.out \
        --alignment-operator ${METHOD} > $OUTDIR/translation_align.txt
```

Where `${METHOD}` is `dtw-medianf` for DTW and `sbaam` for SBAAM.


### 3. SRT Creation

Lastly, build the output SRTs with the command:

```bash
mkdir $OUTDIR/srt

python ${FBK_fairseq}/examples/speech_to_text/make_srt.py $OUTDIR/translation.txt \
        $OUTDIR/translation_align.txt \
        $YAML_DEF \
        $OUTDIR/srt
```

Where `$YAML_DEF` is the YAML file that contains the audio split definitions obtained with SHAS
and `$OUTDIR/srt` is the folder in which the SRT files are written.


## 🔍 Evaluation
Please use the [SubER](https://github.com/apptek/SubER) repository for the `SubER-cased` and `AS-BLEU`
computation.

```bash
suber -H $OUTDIR/srt/* -R $REFERENCE_SRT_DIR/* --metric SubER-cased AS-BLEU
```

The [SubSONAR](https://github.com/hlt-mt/SubSONAR) score, instead, can be computed as:

```bash
declare -A lang_codes ; lang_codes["de"]="deu_Latn" ; lang_codes["es"]="spa_Latn" ; \
        lang_codes["fr"]="fra_Latn" ; lang_codes["it"]="ita_Latn" ; lang_codes["nl"]="nld_Latn" ;
        lang_codes["pt"]="por_Latn" ; lang_codes["ro"]="ron_Latn"

subsonar --srt-files $OUTDIR/srt/* --audio-files $TEST_SET_AUDIOS/* \
        -bs 64 --audio-lang eng --text-lang "${lang_codes["$lang"]}"
```

To evaluate CPL and CPS conformity, run:
```bash
python ${FBK_fairseq}/FBK-fairseq/examples/speech_to_text/scripts/subtitle_compliance.py \
        --srt-file $OUTDIR/srt/*.srt \
        --metrics cpl cps --remove-parenthesis-content
```

## ⭐ Citation
If you use this work, please cite:
```bibtex
@inproceedings{gaido-et-al-2024-sbaam,
  title = {{SBAAM! Eliminating Transcript Dependency in Automatic Subtitling}},
  author = {Gaido, Marco and Papi, Sara and Negri, Matteo and Cettolo, Mauro and Bentivogli, Luisa},
  booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
  year = "2024",
  address = "Bangkok, Thailand",
}
```
