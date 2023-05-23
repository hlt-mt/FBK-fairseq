# Joint Speech Translation and Named Entity Recognition

This README contains basic instructions to reproduce the results
of the paper: [Joint Speech Translation and Named Entity Recognition](https://arxiv.org/abs/2210.11987).

## Training

Below, you can find the training scripts used for our ST systems.
Before training, please follow the preprocessing steps described
[here](SPEECHFORMER.md#preprocessing).

You also need to add the NE open and close tags to the Sentencepiece dictionaries
for the `Inline` approach.

### ST system of the ST+NER Cascade

After defining `data_root` as the folder where you preprocessed the MuST-C and Europarl-ST datasets,
run the following script (it assumes using 4 K80 GPUs, if you are using different number or type
of GPUs you need to update the `--update-frequency` and `--max-tokens` to maximize the VRAM usage
and keep their product multiplied by the number of GPUs to 320,000).

```bash
python train.py $data_root \
	--train-subset train_mustc_notags,train_ep_notags --valid-subset dev_ep_notags \
	--save-dir $st_save_dir \
	--num-workers 2 --max-update 100000 \
	--max-tokens 10000 \
	--user-dir examples/speech_to_text \
	--task speech_to_text_ctc --config-yaml config_st_noterms.yaml  \
	--criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--arch conformer \
	--ctc-encoder-layer 8 --ctc-weight 0.5 --ctc-compress-strategy avg \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 \
	--seed 9 --update-freq 8 --patience 5 --keep-last-epochs 7 \
	--skip-invalid-size-inputs-valid-test --find-unused-parameters
	
```

where `config_st_noterms.yaml` is:

```
bpe_tokenizer:
  bpe: sentencepiece
  sentencepiece_model: YOUR_TGT_SP_MODEL
bpe_tokenizer_src:
  bpe: sentencepiece
  sentencepiece_model: YOUR_SRC_SP_MODEL
input_channels: 1
input_feat_per_channel: 80
sampling_alpha: 1.0
specaugment:
  freq_mask_F: 27
  freq_mask_N: 2
  time_mask_N: 2
  time_mask_T: 100
  time_mask_p: 1.0
  time_wrap_W: 0
transforms:
  '*':
  - utterance_cmvn
  _train:
  - utterance_cmvn
  - specaugment
vocab_filename: YOUR_TGT_TXT_VOCAB
vocab_filename_src: YOUR_SRC_TXT_VOCAB
tags:
 - CARDINAL
 - DATE
 - EVENT
 - FAC
 - GPE
 - LANGUAGE
 - LAW
 - LOC
 - MONEY
 - NORP
 - ORDINAL
 - ORG
 - PERCENT
 - PERSON
 - PRODUCT
 - QUANTITY
 - TIME
 - WORK_OF_ART
```


### Inline Joint ST and NER

First of all, you need to create a new training TSV where each sentence of the transcripts and
translations have been annotated with Deeppavlov (we used the `ner_ontonotes_bert_mult` available in Huggingface).

Then the training script is similar to the previous one, only the training data is different:

```bash
python train.py $data_root \
	--train-subset train_mustc_netagged,train_ep_netagged --valid-subset dev_ep_netagged \
	--save-dir $st_save_dir \
	--num-workers 2 --max-update 100000 \
	--max-tokens 10000 \
	--user-dir examples/speech_to_text \
	--task speech_to_text_ctc --config-yaml config_st_noterms.yaml  \
	--criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--arch conformer \
	--ctc-encoder-layer 8 --ctc-weight 0.5 --ctc-compress-strategy avg \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 \
	--seed 9 --update-freq 8 --patience 5 --keep-last-epochs 7 \
	--skip-invalid-size-inputs-valid-test --find-unused-parameter
```

### Parallel Joint ST and NER

In this case, we use the same training data as the `Inline` approach, but the training script is:

```bash
python train.py $data_root \
	--train-subset train_mustc_netagged,train_ep_netagged --valid-subset dev_ep_netagged \
	--save-dir $st_save_dir \
	--num-workers 2 --max-update 100000 \
	--max-tokens 10000 \
	--user-dir examples/speech_to_text \
	--task speech_to_text_ctc_tagged --config-yaml config_st_noterms.yaml  \
	--criterion ctc_multi_loss --underlying-criterion cross_entropy_with_tags --label-smoothing 0.1 --tags-loss-weight 1.0 \
	--arch conformer_with_tags \
	--ctc-encoder-layer 8 --ctc-weight 0.5 --ctc-compress-strategy avg \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 \
	--seed 9 --update-freq 8 --patience 5 --keep-last-epochs 7 \
	--skip-invalid-size-inputs-valid-test --find-unused-parameters
```

For the `+ NE emb.` variant, add the `--add-tags-embeddings` flag to this script.

## 📍Citation

```bibtex
@inproceedings{gaido-et-al-2023-joint,
title = {{Joint Speech Translation and Named Entity Recognition}},
author = {Gaido, Marco and Papi, Sara and Negri, Matteo and Turchi, Marco},
booktitle = "Proc. of Interspeech 2023",
year = "2023",
address = "Dublin, Ireland"
}
```
