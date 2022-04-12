# Speechformer
This repository contains the code for the preprocessing, training and evaluation steps of the `PlainConvattention` and 
`Speechformer` architectures as well as the pretrained models.

For further details, please refer to the paper: [Speechformer: Reducing Information Loss in Direct Speech Translation](https://arxiv.org/).

## Setup
Clone this repository and install it as explained in the original [Fairseq(-py)](https://github.com/pytorch/fairseq).
For the experiments we used MuST-C (en-de, en-es, en-nl), make sure to [download the corpus](https://ict.fbk.eu/must-c/).

## Preprocessing
Before starting the training, the data has to be preprocessed.
To preprocess the data, run the following command, where
`DATA_ROOT` is the language-specific MuST-C 
directory, by `FAIRSEQ_DIR` is the path to this Fairseq installation and by `MUSTC_SAVE_DIR` is the path where you want to 
save the preprocessed files:

```sh
python ${FAIRSEQ_DIR}/examples/speech_to_text/preprocess_generic.py \
  --data-root ${DATA_ROOT} --wav-dir ${DATA_ROOT}/wav \
  --save-dir ${MUSTC_SAVE_DIR} \
  --task st --src-lang en --tgt-lang ${LANG} \
  --splits train dev tst-COMMON \
  --vocab-type unigram \
  --vocab-size 8000 \
  --src-normalize 
```

⭐️*Pay attention!* ➜ To replicate the experiments of the Speechformer, the source vocabulary size has to be **5000**. You have to run this 
script again changing `--vocab-size 8000` to `--vocab-size 5000`, with the option 
`--no-filterbank-extraction` to avoid the re-computation of the mel-filterbank features. 

##Training 
In the following, there are the scripts for training both `PlainConvattention` and `Speechformer` architectures.

⭐️**Please note** that the training phase of `PlainConvattention` (which corresponds to the encoder pretraining of the 
Speechformer) is *mandatory* to successfully train the `Speechformer` architecture.
###PlainConvattention
To start the training of the `PlainConvattention` architecture, run the following command, where `ST_SAVE_DIR` is the directory in which you 
want to save the trained model and `CONFIG_YAML_NAME` is the name of the .yaml file:
```sh
fairseq-train ${MUSTC_SAVE_DIR} \
        --train-subset train_st_src --valid-subset dev_st_src \
        --save-dir ${ST_SAVE_DIR} \
        --num-workers 8 --max-update 100000 \
        --max-tokens 10000 \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc --config-yaml ${CONFIG_YAML_NAME}.yaml \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 --best-checkpoint-metric loss \
        --arch speechformer_m \
        --ctc-encoder-layer 8 \
        --compressed 4 --compress-kernel-size 8 --stride 1 \
        --shared-layer-kv-compressed --shared-kv-compressed \
        --CNN-first-layer \
        --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
        --warmup-updates 10000 \
        --clip-norm 10.0 \
        --seed 1 --update-freq 16 \
        --skip-invalid-size-inputs-valid-test 
```
The script above is intended to be run on 2 V100 GPUs with 32GB of RAM. In case you have more GPUs, you should divide 
the `--update-freq` parameter accordingly, e.g. if you have 4 GPUs use 8 as `--update-freq`. 
In case your GPUs have lower RAM, you can halve the `--max-tokens` value and duplicate `--update-freq`.

### Speechformer
To start the training of the `Speechformer` arcitecture, the first step is to select only the first part of the
`PlainConvattention` encoder (until the layer to which the CTC is 
applied) by running this command:
```sh
python ${FAIRSEQ_DIR}/examples/speech_to_text/scripts/strip_after_ctc.py \
  --user-dir examples/speech_to_text \
  --model-path ${CHECKPOINT_PATH} --encoder-layer-name speechformer_layer \
  --new-model-path ${STRIPPED_CHECKPOINT_PATH}
```
where `CHECKPOINT_PATH` is the absolute path to your PlainConvattention checkpoint .pt and `STRIPPED_CHECKPOINT_PATH` is the absolute path 
to the new checkpoint .pt generated containing only the first part of the encoder. Also `--num-encoder-layers` and 
`--ctc-encoder-layer` have to be specified if different from our default architecture 
(with values 12 and 8 respectively).

⭐️**Please note** that, to replicate our paper, the checkpoint used are the average 7, as explained in the **Generate** section. 

Then, to start the training, run the following command:
```sh
fairseq-train ${MUSTC_SAVE_DIR} \
        --train-subset train_st_src --valid-subset dev_st_src \
        --save-dir ${ST_SAVE_DIR} \
        --num-workers 8 --max-update 100000 \
        --max-tokens 10000 \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc --config-yaml ${CONFIG_YAML_NAME}.yaml  \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
        --label-smoothing 0.1 --best-checkpoint-metric loss \
        --arch speechformer_m \
        --load-pretrained-encoder-from ${STRIPPED_CHECKPOINT_PATH} \
        --allow-partial-encoder-loading \
        --transformer-after-compression \
        --ctc-encoder-layer 8 \
        --ctc-compress-strategy avg \
        --compressed 4 --compress-kernel-size 8 --stride 1 \
        --shared-layer-kv-compressed --shared-kv-compressed \
        --CNN-first-layer \
        --optimizer adam --lr 1e-3 --lr-scheduler inverse_sqrt \
        --warmup-updates 10000 \
        --clip-norm 10.0 \
        --seed 1 --update-freq 16 \
        --skip-invalid-size-inputs-valid-test
```
and you can use the parameter `--patience` to early stopping the training once the loss does not improve for a certain 
number of epochs (15 in our case).

## Generate
For the generate phase, you first have to *average 7 checkpoints*, among which the middle one is the best checkpoint 
on the validation set (according to the loss) obtained during training.
Run the following command and set `BEST_CKP+3` as the number of your best checkpoint plus 3 to make the average 7 and 
`AVERAGE_CHECKPOINT_NAME` as the name that you want to give to the average checkpoint:
```sh
python ${FAIRSEQ_DIR}/scripts/average_checkpoints.py \
  --inputs ${ST_SAVE_DIR} \
  --output "${ST_SAVE_DIR}/${AVERAGE_CHECKPOINT_NAME}.pt" \
  --num-epoch-checkpoints 7 \
  --checkpoint-upper-bound ${BEST_CKP+3}
```

Then, run the following command to perform the generate:
```sh
fariseq-generate ${MUSTC_SAVE_DIR} \
  --config-yaml ${CONFIG_YAML_NAME}.yaml \
  --gen-subset tst-COMMON_st_src \
  --task speech_to_text_ctc \
  --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
  --user-dir examples/speech_to_text \
  --path ${ST_SAVE_DIR}/${AVERAGE_CHECKPOINT_NAME}.pt \
  --max-tokens 25000 --beam 5 --scoring sacrebleu --no-repeat-ngram-size 5 \
  --results-path ${ST_SAVE_DIR}
```
Note that we set `--max-tokens 25000` since we used a K80 GPU with 12 GB of RAM to generate the output.

## ⭐️PRETRAINED MODELS

Download our vocabulary and yaml files if you want to use our pretrained models:
- [Generic yaml](https://drive.google.com/file/d/1n7yKpoFgixf7XdiJEy2lsJLbkHVzKluf/view?usp=sharing)
- Source: [En](https://drive.google.com/file/d/1GjsvQ6n0C92E2YY8Wpf048rEKDjG1Jrs/view?usp=sharing)
- Target: [De](https://drive.google.com/file/d/1vVwH1oOLuqFQ4-xh6P7ZHj-t8iMEDX1d/view?usp=sharing), 
  [Nl](https://drive.google.com/file/d/1CyX3ANOta2lHrALeGvOCazxVnzfSRfeF/view?usp=sharing),
  [Es](https://drive.google.com/file/d/1apnlOgVyxqhT-2lYiAyW52pwIz05_Krc/view?usp=sharing)
 

Click on the corresponding language pair to download the model:

| Model | --arch | Params | en-de | en-nl | en-es |
|---|---|---|---|---|---|
| Baseline | s2t_transformer_m_fbk |77M| [22.87](https://drive.google.com/file/d/1PMQsu3sIdjjBFYTqYE0JneRCqhiy3CcR/view?usp=sharing) | [27.21](https://drive.google.com/file/d/1Qij9foYi7Vfa0c1pql6eqtLAZnqoo8Zw/view?usp=sharing) | [28.09](https://drive.google.com/file/d/1abd0cwCBz419_IxPLSruEplyuqhNY-0e/view?usp=sharing) |
| Baseline+compress. | s2t_transformer_m_fbk |77M| [22.89](https://drive.google.com/file/d/1W6CsYxOAxkTADLZK3EIHzwMYBat_0VCW/view?usp=sharing) | [26.93](https://drive.google.com/file/d/1o04QxQKhciSNpswoQPublFSQO-71VzaL/view?usp=sharing) | [28.09](https://drive.google.com/file/d/13LS-VItrXeZtRbgfQGl3puxkTxCoD-fD/view?usp=sharing) |
| PlainConvattn | speechformer_m |79M|[23.29](https://drive.google.com/file/d/18qNNbOgPtoEUbyasNNi2ESzCGBzdjLs3/view?usp=sharing) | [27.18](https://drive.google.com/file/d/1cNQaS70TELzkUXTyZ35NU-4pTUMWW60n/view?usp=sharing) | [28.01](https://drive.google.com/file/d/1WBGzQW9nh2eCoVYVNUn7u-j70lAn0VDC/view?usp=sharing) |
| Speechformer | speechformer_m |79M| [23.84](https://drive.google.com/file/d/11oOIwHm16917JC5seH9QyxPtvcvmogEX/view?usp=sharing) | [27.85](https://drive.google.com/file/d/1QYlCj4w_uGXFZBmej2_eoXUtYgtqhrd5/view?usp=sharing) | [28.56](https://drive.google.com/file/d/1fLaU1sS_NLKKrRHGYPh5YmSIDHjzun68/view?usp=sharing) |

Remember that the results in our paper are the average BLEU score of 3 runs, here you can download the checkpoint a of a single run.

# Citation

Please cite as:

``` bibtex
@inproceedings{papi2021speechformer,
  title = {{Speechformer: Reducing Information Loss in Direct Speech Translation}},
  author = {Papi, Sara and Gaido, Marco and Negri, Matteo and Turchi, Marco},
  booktitle = {Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2021},
}
```