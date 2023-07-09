# Direct Speech Translation for Automatic Subtitling

Code and models for the paper "[Direct Speech Translation for Automatic Subtitling](https://arxiv.org/pdf/2209.13192.pdf)" published at TACL 2023.

## 📌 Pretrained models 
### 📎 MuST-Cinema (multilingual)
- **English > {Dutch, French, German, Italian, Portuguese, Romanian, Spanish}**: [model.pt](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EdWDn-yGs71Jj2wSvFdT7rgBBHbFxAxejdylKH2O0-mZGQ?e=cNPhQa) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EZHmdY4GH-FHrUfKoKzy8AAB0iORJQ3nLCxv6P0g_1xUWA?e=0eEvYY) | [src_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/ERs15CM9DDJHnrgtm1aTNggBNIVUwxky_RShW-BygBSlSQ?e=fM9vEi) | [src_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EbFH9eQvkz5CicnzWTZL7aIB8n1VWCzlmy5RHj6KKuIJUw?e=15JKFC) | [tgt_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/Ef-ZRJ2QOY1Ev9fgtA6QdeYB10cGkvQJdJ2kq0A9Nky-QA?e=5WqVd8) | [tgt_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EWc52NhLTjpDnSdOPDDJZlMBwA0ZFcEkc9a3glhM4FNNig?e=u52Ih8)
### 📎 Unconstrained (or Constrained setting of IWSLT 2023)
- **English > German**: [model.pt](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EXeXzv9YYOJFkH5uqGz9vWgBnLTA0F1Z8ydKDfyIf_Pcsw?e=dbeAOz) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EawpoY0LAtlGtC8vwnP2xygBmunq3knlocG7pVcHI8KryQ?e=RqOmfU) | [src_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EbRppHs78_FJliK3f8bU4fgBSBYO6Nlibx7vvO-JMH5IVQ?e=U7D9vo) | [src_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EYAlO1Ssm7JFg82emEtH2bABxcEwr4w7Ea5XBjEy-0VGdA?e=5wDZKh) | [tgt_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EWHbDsVUUylCpLGw_zG71vcB7awnkd0H6k-_rIE1ycfD-A?e=s2j1gg) | [tgt_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/ETBLNiRa57FOmeDma8yk7BEB8AYvqUFTAQ5ntvUzsNIOZw?e=Lg3mFa)
- **English > Spanish**: [model.pt](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/ETuG9gzjay9LicABTT2vDzkBiaV5nz4vT70FEHbMzm6DIQ?e=Yw0DEf) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EYoX1hagGuRNg11dd5PHRc0BUGJxkAsBoFF7Iiwypk_AKg?e=FZ1WC9) | [src_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EW2pYCbiyOtLv6O9jVhOaHwBifoSkPMuz8JtiHGII5nmvw?e=VD33sB) | [src_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/ES8mXoAJLSlGm3gd_T7ssAIBhBvHWqAS5LmmPhLoVPyfEw?e=LNN8Vp) | [tgt_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EQnbdEtrN15EiRvkR0RDFYgB6kNkYKN-88TRey0xBLQroQ?e=AywcAm) | [tgt_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/ERzTk5UIGOtFlI1BZh4kO6EBphS4nq3NYv2R3ygI5yJHxg?e=AfLkrf)

## 📍 Preprocess and Setup
Clone this repository and install it as explained in the original [Fairseq(-py)](https://github.com/pytorch/fairseq).

Download all the corpora listed in our paper and preprocess them as explained [here](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/SPEECHFORMER.md#preprocessing). 

## 🏃 Training
To train the model from scratch, please follow the two steps below.

### 🔘 Speech Translation Pre-training
First, a speech translation pre-training is performed using all the corpora listed in our paper, 
included MuST-Cinema from which we removed `<eob>` and `<eol>` from the textual parts (transcripts and translation).

Run the following code by setting `${FBK_fairseq}` as the folder containing this repository,
`${DATA_ROOT}` as the folder containing the preprocessed datasets, 
`${SPLIT_LIST}` as the comma separated list of the training ST datasets (e.g. `mustc_train,europarl_train,...`),
`${MUSTCINEMA_DEV}` as the split name of the MuST-Cinema dev set from which `<eob>` and `<eol>` have been removed,
`${ST_SAVE_DIR}` as the directory in which the checkpoints will be saved,
`${CONFIG_YAML}` as the path to the yaml file generated after preprocessing.

This script is intended for 4 NVIDIA A100 40GB, please set `--max-tokens` and `--update-freq` accordingly with 
your hardware, so that `number of GPUs * max_tokens * update_freq = 320,000`.

```bash
python ${FBK_fairseq}/train.py ${DATA_ROOT} \
        --train-subset ${SPLIT_LIST} \
        --valid-subset ${MUSTCINEMA_DEV} \
        --save-dir ${ST_SAVE_DIR} \
        --num-workers 2 --max-update 100000 \
        --save-interval-updates 1000 \
        --max-tokens 40000 --adam-betas '(0.9, 0.98)' \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc --config-yaml ${CONFIG_YAML} \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch conformer \
        --ctc-encoder-layer 8 --ctc-weight 0.5 --ctc-compress-strategy avg \
        --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
        --warmup-updates 25000 --patience 15 \
        --clip-norm 10.0 \
        --seed 1 --update-freq 2 \
        --skip-invalid-size-inputs-valid-test \
        --log-format simple >> ${ST_SAVE_DIR}/train.log
```

### 🔘 Subtitling Fine-tuning
Second, we fine-tune the previous model using the previously preprocessed data but with transcripts and translations
containing `<eob>` and `<eol>`. MuST-Cinema already contains the subtitle segmentation markers, 
while all the other datasets have to be segmented into subtitles 
[using the multimodal segmenter](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/SUBTITLE_SEGMENTER_AACL2022.md).

Please average the checkpoints of the ST pre-trained as explained 
[here](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/SPEECHFORMER.md#generate)
and copy it with the name `checkpoint_last.pt` in the `${SUB_SAVE_DIR}` folder.

Run the following code by setting 
`${SPLIT_LIST}` as the comma separated list of the training ST datasets containing `<eob>` and `<eol>` 
(e.g. `mustc_sub_train,europarl_sub_train,...`),
`${MUSTCINEMA_DEV}` as the split name of the original MuST-Cinema dev set containing `<eob>` and `<eol>`,
`${SUB_SAVE_DIR}` as the folder in which to save the checkpoints for the final model.

```bash
python ${FBK_fairseq}/train.py ${DATA_ROOT} \
        --train-subset ${SPLIT_LIST} \
        --valid-subset ${MUSTCINEMA_DEV} \
        --save-dir ${SUB_SAVE_DIR} \
        --num-workers 1 --max-update 100000 --save-interval-updates 1000 \
        --max-tokens 40000 --adam-betas '(0.9, 0.98)' \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc --config-yaml ${CONFIG_YAML} \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch conformer \
        --ctc-encoder-layer 8 --ctc-weight 0.5 --ctc-compress-strategy avg \
        --optimizer adam --lr 1e-3 --lr-scheduler fixed \
        --patience 10 \
        --clip-norm 10.0 \
        --seed 1 --update-freq 2 \
        --skip-invalid-size-inputs-valid-test \
        --log-format simple >> ${ST_SAVE_DIR}/train.log
```
Then, average the checkpoints as mentioned above to obtain the final model `checkpoint_avg7.pt`. 

## 📺 Generation
Please use [SHAS](https://github.com/mt-upc/SHAS) to generate the automatic segmentation files for 
[MuST-Cinema](https://ict.fbk.eu/must-cinema/) test set, 
[EC Short Clips](https://mt.fbk.eu/ec-short-clips/), and 
[EuroParl Interviews](https://mt.fbk.eu/europarl-interviews/) 
as we do in our paper and preprocess them.

To generate the srt files, run the below script by setting: 
`${DATA_ROOT}` as the folder containing the prepreocessed test set,
`${SPLIT}` as the name of the preprocessed test set,
`${CONFIG_YAML}` as the path to the yaml file of the model,
`${MODEL}` as the path to the model checkpoint in pt format,
`${YAML_SEGM}` as the path to the yaml containing the automatic segmentation obtained by SHAS 
which has been also used during the preprocessing,
`${SRT_DIR}` as the output folder that will contain the generated srt.

The script performs the generation process into 4 steps. This was done to ease
the experimentation with different methods, although inefficient.
An efficient implementation could perform the full generation with a single
forward on the direct ST model.

```bash
DATA_ROOT=$1
SPLIT=$2
CONFIG_YAML=$3
MODEL=$4
YAML_SEGM=$5
SRTDIR=$6

data_tmp_dir=$(mktemp -d)
mkdir -p $SRTDIR

# Generates the output subtitles (translations with <eol> and <eob>)
# with the autoregressive decoder
python ${FBK_fairseq}/FBK-fairseq/fairseq_cli/generate.py $DATA_ROOT \
        --user-dir examples/speech_to_text --config-yaml $CONFIG_YAML \
        --gen-subset $SPLIT  \
        --max-tokens 16000 --unkpen 10000 --beam 5 \
        --model-overrides "{'batch_unsafe_relative_shift': False}" \
        --max-source-positions 16000 --max-target-positions 1000 \
        --task speech_to_text_ctc --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy --no-repeat-ngram-size 5 \
        --path $MODEL > $data_tmp_dir/translation.out

grep "^D-" $data_tmp_dir/translation.out | cut -c3- | sort -k 1n | cut -f3 > $data_tmp_dir/translation.txt

# Generates the captions (transcripts with <eol> and <eob>) using
# the CTC predictions
python ${FBK_fairseq}/FBK-fairseq/fairseq_cli/generate.py $DATA_ROOT \
        --user-dir examples/speech_to_text --config-yaml $CONFIG_YAML \
        --gen-subset  $SPLIT \
        --max-tokens 16000 --unkpen 10000 --beam 5 \
        --model-overrides "{'batch_unsafe_relative_shift': False}"  \
        --max-source-positions 16000 --max-target-positions 1000 \
        --task speech_to_text_ctcgen --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy --no-repeat-ngram-size 5 \
        --path $MODEL --lenpen 0.0 > $data_tmp_dir/transcript.out

grep "^D-" $data_tmp_dir/transcript.out | cut -c3- | sort -k 1n | cut -f3 > $data_tmp_dir/transcript.txt

# Runs the CTC segmentation to align the generated transcripts with the source
# audio, hence obtaining the estimated timestamps at block level
python ${FBK_fairseq}/FBK-fairseq/examples/speech_to_text/scripts/ctc_align.py $DATA_ROOT \
        --user-dir examples/speech_to_text --config-yaml $CONFIG_YAML \
        --gen-subset $SPLIT \
        --max-tokens 16000 --beam 5 \
        --model-overrides "{'batch_unsafe_relative_shift': False}" \
        --max-source-positions 16000 --max-target-positions 1000 \
        --split-tokens "<eob>" --feature-duration 0.04 \
        --task speech_to_text_ctc \
        --criterion ctc_multi_loss --underlying-criterion cross_entropy \
        --path $MODEL --text-file $data_tmp_dir/transcript.txt > $data_tmp_dir/transcript_align.out

grep "^SEGM-" $data_tmp_dir/transcript_align.out | cut -c6- | sort -k 1n | cut -f2 > $data_tmp_dir/transcript_align.txt

# Projects the caption timestamps onto the subtitling blocks with the Levenshtein method
python ${FBK_fairseq}/FBK-fairseq/examples/speech_to_text/scripts/target_from_source_timestamp_levenshtein.py \
        $data_tmp_dir/transcript.txt \
        $data_tmp_dir/transcript_align.txt \
        $data_tmp_dir/translation.txt \
        $data_tmp_dir/translation_align.txt

# Creates the SRT files
python ${FBK_fairseq}/FBK-fairseq/examples/speech_to_text/make_srt.py \
        $data_tmp_dir/translation.txt \
        $data_tmp_dir/translation_align.txt \
        $YAML_SEGM \
        $SRTDIR

rm -rf $data_tmp_dir
```

## 🔍 Evaluation
Please use [SubER](https://github.com/apptek/SubER) repository for the `SubER-cased` computation.

To evaluate BLEU (BLEUnb) and Sigma, please install [EvalSub](https://github.com/fyvo/EvalSubtitle) and 
[mwerSegmenter](https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz), and 
run the following code by setting
`${REF_SRTDIR}` as the folder containing the reference srt files,
`${MWER_DIR}` as the folder containing the mwerSegmenter,
`${EVALSUB_DIR}` as the folder containing the EvalSub, and
`${OUT_FILE}` as the path in which to save the EvalSub output.

```bash
# These first 4 commands should be skipped for MuST-Cinema.
# For MuST-Cinema, use as reference the amara.$lang files
# instead of generating the text files from the SRTs.
cat ${SRTDIR}/*.srt > ${SRTDIR}/hyp.srt
cat ${REF_SRTDIR}/*.srt > ${REF_SRTDIR}/ref.srt
python from_srt_to_blocks.py ${SRTDIR}/hyp.srt
python from_srt_to_blocks.py ${REF_SRTDIR}/ref.srt

${MWER_DIR}/mwerSegmenter \
    -mref ${REF_SRTDIR}/ref.srt.blocks \
    -hypfile ${SRTDIR}/hyp.srt.blocks 
mv __segments ${SRTDIR}/hyp.srt.blocks.resegm; 

python ${EVALSUB_DIR}/evalsub_main.py -a -e2e \
    -ref ${REF_SRTDIR}/ref.srt.blocks \
    -sys ${SRTDIR}/hyp.srt.blocks.resegm \
    -res ${OUT_FILE}
```

To evaluate CPL and CPS conformity, run:
```bash
python ${FBK_fairseq}/FBK-fairseq/examples/speech_to_text/scripts/subtitle_compliance.py \
        --srt-file ${SRTDIR}/*.srt \
        --metrics cpl cps --remove-parenthesis-content
```

## ⭐ Citation
If you use this work, please cite:
```bibtex
@article{papi2023directsub,
  title={Direct Speech Translation for Automatic Subtitling},
  author={Papi, Sara and Gaido, Marco and Karakanta, Alina and Cettolo, Mauro and Negri, Matteo and Turchi, Marco},
  journal={Transactions of the Association for Computational Linguistics},
  year={2023}
}
```
