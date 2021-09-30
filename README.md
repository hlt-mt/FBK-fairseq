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
python ${FAIRSEQ_DIR}/examples/speech_to_text/strip_after_ctc.py \
  --user-dir examples/speech_to_text \
  --model-path ${CHECKPOINT_PATH} \
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


Below, there is the original Fairseq README file.

--------------------------------------------------------------------------------

<p align="center">
  <img src="docs/fairseq_logo.png" width="150">
  <br />
  <br />
  <a href="https://github.com/pytorch/fairseq/blob/master/LICENSE"><img alt="MIT License" src="https://img.shields.io/badge/license-MIT-blue.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/releases"><img alt="Latest Release" src="https://img.shields.io/github/release/pytorch/fairseq.svg" /></a>
  <a href="https://github.com/pytorch/fairseq/actions?query=workflow:build"><img alt="Build Status" src="https://github.com/pytorch/fairseq/workflows/build/badge.svg" /></a>
  <a href="https://fairseq.readthedocs.io/en/latest/?badge=latest"><img alt="Documentation Status" src="https://readthedocs.org/projects/fairseq/badge/?version=latest" /></a>
</p>

--------------------------------------------------------------------------------

Fairseq(-py) is a sequence modeling toolkit that allows researchers and
developers to train custom models for translation, summarization, language
modeling and other text generation tasks.

We provide reference implementations of various sequence modeling papers:

<details><summary>List of implemented papers</summary><p>

* **Convolutional Neural Networks (CNN)**
  + [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/conv_lm/README.md)
  + [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
  + [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
  + [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
  + [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
* **LightConv and DynamicConv models**
  + [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
* **Long Short-Term Memory (LSTM) networks**
  + Effective Approaches to Attention-based Neural Machine Translation (Luong et al., 2015)
* **Transformer (self-attention) networks**
  + Attention Is All You Need (Vaswani et al., 2017)
  + [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
  + [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
  + [Adaptive Input Representations for Neural Language Modeling (Baevski and Auli, 2018)](examples/language_model/README.adaptive_inputs.md)
  + [Lexically constrained decoding with dynamic beam allocation (Post & Vilar, 2018)](examples/constrained_decoding/README.md)
  + [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context (Dai et al., 2019)](examples/truncated_bptt/README.md)
  + [Adaptive Attention Span in Transformers (Sukhbaatar et al., 2019)](examples/adaptive_span/README.md)
  + [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
  + [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
  + [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
  + [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md )
  + [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
  + [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
  + [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
  + [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
  + [Generating Medical Reports from Patient-Doctor Conversations Using Sequence-to-Sequence Models (Enarvi et al., 2020)](examples/pointer_generator/README.md)
  + [Linformer: Self-Attention with Linear Complexity (Wang et al., 2020)](examples/linformer/README.md)
  + [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
  + [Deep Transformers with Latent Depth (Li et al., 2020)](examples/latent_depth/README.md)
* **Non-autoregressive Transformers**
  + Non-Autoregressive Neural Machine Translation (Gu et al., 2017)
  + Deterministic Non-Autoregressive Neural Sequence Modeling by Iterative Refinement (Lee et al. 2018)
  + Insertion Transformer: Flexible Sequence Generation via Insertion Operations (Stern et al. 2019)
  + Mask-Predict: Parallel Decoding of Conditional Masked Language Models (Ghazvininejad et al., 2019)
  + [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
* **Finetuning**
  + [Better Fine-Tuning by Reducing Representational Collapse (Aghajanyan et al. 2020)](examples/rxf/README.md)

</p></details>

### What's New:

* December 2020: [GottBERT model and code released](examples/gottbert/README.md)
* November 2020: Adopted the [Hydra](https://github.com/facebookresearch/hydra) configuration framework
  * [see documentation explaining how to use it for new and existing projects](docs/hydra_integration.md)
* November 2020: [fairseq 0.10.0 released](https://github.com/pytorch/fairseq/releases/tag/v0.10.0)
* October 2020: [Added R3F/R4F (Better Fine-Tuning) code](examples/rxf/README.md)
* October 2020: [Deep Transformer with Latent Depth code released](examples/latent_depth/README.md)
* October 2020: [Added CRISS models and code](examples/criss/README.md)
* September 2020: [Added Linformer code](examples/linformer/README.md)
* September 2020: [Added pointer-generator networks](examples/pointer_generator/README.md)
* August 2020: [Added lexically constrained decoding](examples/constrained_decoding/README.md)
* August 2020: [wav2vec2 models and code released](examples/wav2vec/README.md)
* July 2020: [Unsupervised Quality Estimation code released](examples/unsupervised_quality_estimation/README.md)

<details><summary>Previous updates</summary><p>

* May 2020: [Follow fairseq on Twitter](https://twitter.com/fairseq)
* April 2020: [Monotonic Multihead Attention code released](examples/simultaneous_translation/README.md)
* April 2020: [Quant-Noise code released](examples/quant_noise/README.md)
* April 2020: [Initial model parallel support and 11B parameters unidirectional LM released](examples/megatron_11b/README.md)
* March 2020: [Byte-level BPE code released](examples/byte_level_bpe/README.md)
* February 2020: [mBART model and code released](examples/mbart/README.md)
* February 2020: [Added tutorial for back-translation](https://github.com/pytorch/fairseq/tree/master/examples/backtranslation#training-your-own-model-wmt18-english-german)
* December 2019: [fairseq 0.9.0 released](https://github.com/pytorch/fairseq/releases/tag/v0.9.0)
* November 2019: [VizSeq released (a visual analysis toolkit for evaluating fairseq models)](https://facebookresearch.github.io/vizseq/docs/getting_started/fairseq_example)
* November 2019: [CamemBERT model and code released](examples/camembert/README.md)
* November 2019: [BART model and code released](examples/bart/README.md)
* November 2019: [XLM-R models and code released](examples/xlmr/README.md)
* September 2019: [Nonautoregressive translation code released](examples/nonautoregressive_translation/README.md)
* August 2019: [WMT'19 models released](examples/wmt19/README.md)
* July 2019: fairseq relicensed under MIT license
* July 2019: [RoBERTa models and code released](examples/roberta/README.md)
* June 2019: [wav2vec models and code released](examples/wav2vec/README.md)

</p></details>

### Features:

* multi-GPU training on one machine or across multiple machines (data and model parallel)
* fast generation on both CPU and GPU with multiple search algorithms implemented:
  + beam search
  + Diverse Beam Search ([Vijayakumar et al., 2016](https://arxiv.org/abs/1610.02424))
  + sampling (unconstrained, top-k and top-p/nucleus)
  + [lexically constrained decoding](examples/constrained_decoding/README.md) (Post & Vilar, 2018)
* [gradient accumulation](https://fairseq.readthedocs.io/en/latest/getting_started.html#large-mini-batch-training-with-delayed-updates) enables training with large mini-batches even on a single GPU
* [mixed precision training](https://fairseq.readthedocs.io/en/latest/getting_started.html#training-with-half-precision-floating-point-fp16) (trains faster with less GPU memory on [NVIDIA tensor cores](https://developer.nvidia.com/tensor-cores))
* [extensible](https://fairseq.readthedocs.io/en/latest/overview.html): easily register new models, criterions, tasks, optimizers and learning rate schedulers
* [flexible configuration](docs/hydra_integration.md) based on [Hydra](https://github.com/facebookresearch/hydra) allowing a combination of code, command-line and file based configuration

We also provide [pre-trained models for translation and language modeling](#pre-trained-models-and-examples)
with a convenient `torch.hub` interface:

``` python
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model')
en2de.translate('Hello world', beam=5)
# 'Hallo Welt'
```

See the PyTorch Hub tutorials for [translation](https://pytorch.org/hub/pytorch_fairseq_translation/)
and [RoBERTa](https://pytorch.org/hub/pytorch_fairseq_roberta/) for more examples.

# Requirements and Installation

* [PyTorch](http://pytorch.org/) version >= 1.5.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)
* **To install fairseq** and develop locally:

``` bash
git clone https://github.com/pytorch/fairseq
cd fairseq
pip install --editable ./

# on MacOS:
# CFLAGS="-stdlib=libc++" pip install --editable ./

# to install the latest stable release (0.10.0)
# pip install fairseq==0.10.0
```

* **For faster training** install NVIDIA's [apex](https://github.com/NVIDIA/apex) library:

``` bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" \
  --global-option="--deprecated_fused_adam" --global-option="--xentropy" \
  --global-option="--fast_multihead_attn" ./
```

* **For large datasets** install [PyArrow](https://arrow.apache.org/docs/python/install.html#using-pip): `pip install pyarrow`
* If you use Docker make sure to increase the shared memory size either with `--ipc=host` or `--shm-size`
 as command line options to `nvidia-docker run` .

# Getting Started

The [full documentation](https://fairseq.readthedocs.io/) contains instructions
for getting started, training new models and extending fairseq with new model
types and tasks.

# Pre-trained models and examples

We provide pre-trained models and pre-processed, binarized test sets for several tasks listed below,
as well as example training and evaluation commands.

* [Translation](examples/translation/README.md): convolutional and transformer models are available
* [Language Modeling](examples/language_model/README.md): convolutional and transformer models are available

We also have more detailed READMEs to reproduce results from specific papers:

* [Cross-lingual Retrieval for Iterative Self-Supervised Training (Tran et al., 2020)](examples/criss/README.md)
* [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations (Baevski et al., 2020)](examples/wav2vec/README.md)
* [Unsupervised Quality Estimation for Neural Machine Translation (Fomicheva et al., 2020)](examples/unsupervised_quality_estimation/README.md)
* [Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)](examples/quant_noise/README.md)
* [Neural Machine Translation with Byte-Level Subwords (Wang et al., 2020)](examples/byte_level_bpe/README.md)
* [Multilingual Denoising Pre-training for Neural Machine Translation (Liu et at., 2020)](examples/mbart/README.md)
* [Reducing Transformer Depth on Demand with Structured Dropout (Fan et al., 2019)](examples/layerdrop/README.md)
* [Jointly Learning to Align and Translate with Transformer Models (Garg et al., 2019)](examples/joint_alignment_translation/README.md)
* [Levenshtein Transformer (Gu et al., 2019)](examples/nonautoregressive_translation/README.md)
* [Facebook FAIR's WMT19 News Translation Task Submission (Ng et al., 2019)](examples/wmt19/README.md)
* [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., 2019)](examples/roberta/README.md)
* [wav2vec: Unsupervised Pre-training for Speech Recognition (Schneider et al., 2019)](examples/wav2vec/README.md)
* [Mixture Models for Diverse Machine Translation: Tricks of the Trade (Shen et al., 2019)](examples/translation_moe/README.md)
* [Pay Less Attention with Lightweight and Dynamic Convolutions (Wu et al., 2019)](examples/pay_less_attention_paper/README.md)
* [Understanding Back-Translation at Scale (Edunov et al., 2018)](examples/backtranslation/README.md)
* [Classical Structured Prediction Losses for Sequence to Sequence Learning (Edunov et al., 2018)](https://github.com/pytorch/fairseq/tree/classic_seqlevel)
* [Hierarchical Neural Story Generation (Fan et al., 2018)](examples/stories/README.md)
* [Scaling Neural Machine Translation (Ott et al., 2018)](examples/scaling_nmt/README.md)
* [Convolutional Sequence to Sequence Learning (Gehring et al., 2017)](examples/conv_seq2seq/README.md)
* [Language Modeling with Gated Convolutional Networks (Dauphin et al., 2017)](examples/language_model/README.conv.md)

# Join the fairseq community

* Twitter: https://twitter.com/fairseq
* Facebook page: https://www.facebook.com/groups/fairseq.users
* Google group: https://groups.google.com/forum/#!forum/fairseq-users

# License

fairseq(-py) is MIT-licensed.
The license applies to the pre-trained models as well.

# Citation

Please cite as:

``` bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```
