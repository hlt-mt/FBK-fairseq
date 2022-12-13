# Multimodal Subtitle Segmenter (AACL 2022)

Code for the paper: 
[Dodging the Data Bottleneck: Automatic Subtitling with Automatically Segmented ST Corpora](https://aclanthology.org/2022.aacl-short.59.pdf).

## Pre-trained multimodal model
We here release the multilingual multimodal model with parallel attention (Figure 1) used for the paper:
- [model_checkpoint](https://drive.google.com/file/d/1ACzTPG3UgK173cPXIRsz1gp8NldRw4uV/view?usp=share_link) | [config.yaml](https://drive.google.com/file/d/1Yttj2QnSPJnrgFqwbRV-wPVifMsE8pWd/view?usp=share_link) | [spm_model](https://drive.google.com/file/d/1KiuTxVUFsIP9-K3Hc_ggubDp-FW_-aKh/view?usp=share_link) | [fairseq_vocabulary](https://drive.google.com/file/d/1wJtiXzIDQMdNOeL0GjlgdMisqsger0qi/view?usp=share_link)

## Preprocessing
Preprocess the [MuST-Cinema](https://ict.fbk.eu/must-cinema/) dataset as already explained 
[here](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/SPEECHFORMER.md#preprocessing).
Then, run the following code:
```bash
for subset in train dev amara; do
        cut -f 5 ${DATA_ROOT}/en-${LANG}/${subset}_st_src.tsv > \
         ${DATA_ROOT}/en-${LANG}/${subset}.${lang}.multimod
        sed 's/<eob>//g; s/<eol>//g; s/  / /g; s/^ //g; s/ $//g' \
         ${DATA_ROOT}/en-${LANG}/${subset}.${LANG}.multimod > \
         ${DATA_ROOT}/en-${LANG}/${subset}.${LANG}.multimod.unsegm
        paste ${DATA_ROOT}/en-${LANG}/${subset}_st_src.tsv \
        ${DATA_ROOT}/en-${LANG}/${subset}.${LANG}.multimod.unsegm \
        | cut -f 1,2,3,5,6,7 > ${DATA_ROOT}/en-${LANG}/${subset}_multi_segm.tsv

        sed -i '1s/tgt_text$/src_text/g' ${DATA_ROOT}/en-${LANG}/${subset}_multi_segm.tsv
        done
```
where `DATA_ROOT` is the folder containing the preprocessed data, `LANG` is the language 
(en, de, fr, it for train, dev, and amara sets and es, nl only for the amara set).
Lastly, add the target language as a tsv column to enable Fairseq-ST multiligual training/inference for each subset and 
for each language:
```bash
awk 'NR==1 {printf("%s\t%s\n", $0, "tgt_lang")}  NR>1 {printf("%s\t%s\n", $0, "'"${LANG}"'")}' \
  ${DATA_ROOT}/en-${LANG}/${subset}_multi_segm.tsv > ${DATA_ROOT}/${subset}_${LANG}_multi.tsv
```

To generate a unique SentencePiece model and, consequently, vocabulary for all the training languages 
(as we do in our paper), run the script below:
```bash
python ${FBK_fairseq}/examples/speech_to_text/scripts/gen_multilang_spm_vocab.py \
  --data-root ${DATA_ROOT} --save-dir ${DATA_ROOT} \
  --langs en,de,fr,it --splits train_en_multi,train_de_multi,train_fr_multi,train_it_multi \
  --vocab-type unigram --vocab-size 10000
```

## Training
To train the multilingual multimodal model with parallel attention, run the code below:
```bash
python ${FBK_fairseq}/train.py ${DATA_ROOT} \
        --train-subset train_de_multi,train_en_multi,train_fr_multi,train_it_multi \
        --valid-subset dev_de_multi,dev_en_multi,dev_fr_multi,dev_it_multi \
        --save-dir ${SAVE_DIR} \
        --num-workers 2 --max-update 200000 \
        --max-tokens 40000 \
        --user-dir examples/speech_to_text \
        --task speech_to_text_multimodal --config-yaml ${CONFIG_YAML}  \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 \
        --arch s2t_transformer_dual_encoder_s \
        --ctc-encoder-layer 8 --ctc-compress-strategy avg --ctc-weight 0.5 \
        --context-encoder-layers 12 --decoder-layers 3 \
        --context-dropout 0.3 --context-ffn-embed-dim 1024 \
        --share-encoder-decoder-embed \
        --context-decoder-attention-type parallel \
        --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
        --warmup-updates 10000 \
        --clip-norm 10.0 \
        --seed 1 --update-freq 4 \
        --patience 15 \
        --ignore-prefix-size 1 \
        --skip-invalid-size-inputs-valid-test \
        --log-format simple --find-unused-parameters
```
where `FBK_fairseq` is the folder of our repository, `DATA_ROOT` is the folder containing the preprocessed data, 
`SAVE_DIR` is the folder in which to save the checkpoints of the model, `CONFIG_YAML` is the path to the config 
yaml file.

This training setup is intended for 2 NVIDIA A40 48GB, please adjust `--max-tokens` and `--update-freq` such as 
`max_tokens * update_freq * number of GPUs used for training = 320,000`.

## Generation
First, average the checkpoint as already explained in our repository 
[here](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/SPEECHFORMER.md#generate).

Second, run the code below:
```bash
python ${FBK_fairseq}/generate.py ${DATA_ROOT} \
      --config-yaml ${CONFIG_YAML} --gen-subset amara_${LANG}_multi \
      --task speech_to_text_multimodal \
      --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
      --user-dir examples/speech_to_text \
      --path ${SAVE_DIR}/${CHECKPOINT_FILENAME} \
      --max-tokens 25000 --beam 5 --scoring sacrebleu \
      --results-path ${SAVE_DIR}
```
where `LANG` is the language selected for inference (in our paper, Spanish or es and Dutch or nl for the zero-shot 
results) and `CHECKPOINT_FILENAME` is the file containing the average of the checkpoints of the previous step.

Please use [sacrebleu](https://github.com/mjpost/sacrebleu) to obtain BLEU scores and 
[EvalSubtitle](https://github.com/fyvo/EvalSubtitle) to obtain Sigma and CPL.
 
## Citation
```bibtex
@inproceedings{papi-etal-2022-dodging,
    title = "Dodging the Data Bottleneck: Automatic Subtitling with Automatically Segmented {ST} Corpora",
    author = "Papi, Sara  and
      Karakanta, Alina  and
      Negri, Matteo  and
      Turchi, Marco",
    booktitle = "Proceedings of the 2nd Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 12th International Joint Conference on Natural Language Processing (Volume 2: Short Papers)",
    month = nov,
    year = "2022",
    address = "Online only",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.aacl-short.59",
    pages = "480--487",
    }
```
