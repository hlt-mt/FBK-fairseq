# Prepending or Cross-Attention for Speech-to-Text? An Empirical Comparison (NAACL 2025)

This README contains the instructions to replicate the training and evaluation of the models in the paper
[Prepending or Cross-Attention for Speech-to-Text? An Empirical Comparison](https://arxiv.org/abs/2501.02370)
published at NAACL 2025.

## Training

Below we list the scripts used in our experiments. The scripts were executed using
2 A100 GPUs with 64GB of VRAM. In case of a different environment (e.g., a GPU with less VRAM)
you need to adapt `--max-tokens` (which controls the mini-batch size on a single GPU)
and `--update-freq`, so that  `number of GPUs * max_tokens * update_freq = 320,000`.

### Cross-attention encoder-decoder

The Transformer encoder-decoder with cross-attention (line 1 of Table 1 in the paper)
has been trained using:

```shell
python fbk-fairseq/train.py $data_root \
	--train-subset $train_tsv --valid-subset $dev_tsv --config-yaml $config \
	--save-dir $save_dir --user-dir fbk-fairseq/examples/speech_to_text \
	--task speech_to_text_ctc --criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 \
	--arch s2t_transformer_fbk \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
	--seed 1 --skip-invalid-size-inputs-valid-test \
	--update-freq 4 --max-tokens 40000 --num-workers 4 \
	--max-update 100000 --patience 10 --keep-last-epochs 12 \
	--log-format simple >> $save_dir/train.log 2> $save_dir/train.err
```

Similarly, the Conformer version with CTC auxiliary loss (line 4 of Table 1)
was trained with:

```shell
python fbk-fairseq/train.py $data_root \
	--train-subset $train_tsv --valid-subset $dev_tsv --config-yaml $config \
	--save-dir $save_dir --user-dir fbk-fairseq/examples/speech_to_text \
	--task speech_to_text_ctc --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 --ctc-encoder-layer 8 --ctc-weight 0.5 \
	--arch conformer \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
	--seed 1 --skip-invalid-size-inputs-valid-test \
	--update-freq 4 --max-tokens 40000 --num-workers 4 \
	--max-update 100000 --patience 10 --keep-last-epochs 12 \
	--log-format simple >> $save_dir/train.log 2> $save_dir/train.err
```

And to enable CTC compression (line 4.1 of Table 1), add to this command `--ctc-compress-strategy avg`.

### Decoder-prepending

The decoder-prepending models (line 2 of Table 1) have been trained with:

```shell
python fbk-fairseq-dev/train.py $data_root \
	--train-subset $train_tsv --valid-subset $dev_tsv --config-yaml $config \
	--save-dir $save_dir --user-dir fbk-fairseq/examples/speech_to_text \
	--task speech_to_text_ctc --criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 \
	--arch s2tlm_transformer_fbk --encoder-layers 12 --decoder-layers 6 --causal-prompt-mask \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
	--seed 1 --skip-invalid-size-inputs-valid-test \
	--update-freq 4 --max-tokens 40000 --num-workers 4 \
	--max-update 100000 --patience 10 --keep-last-epochs 12 \
	--log-format simple >> $save_dir/train.log 2> $save_dir/train.err
```

To train the version without causal masking in the speech features, remove `--causal-prompt-mask`.

The Conformer version with CTC auxiliary loss (line 5 of Table 1) was trained with:

```shell
python fbk-fairseq-dev/train.py $data_root \
	--train-subset $train_tsv --valid-subset $dev_tsv --config-yaml $config \
	--save-dir $save_dir --user-dir fbk-fairseq/examples/speech_to_text \
	--task speech_to_text_ctc --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 --ctc-encoder-layer 8 --ctc-weight 0.5 \
	--arch s2tlm_conformer --encoder-layers 12 --decoder-layers 6 --causal-prompt-mask \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
	--seed 1 --skip-invalid-size-inputs-valid-test \
	--update-freq 4 --max-tokens 40000 --num-workers 4 \
	--max-update 100000 --patience 10 --keep-last-epochs 12 \
	--log-format simple >> $save_dir/train.log 2> $save_dir/train.err
```

And, as in the previous case, CTC compression (line 5.1) is obtained by adding  `--ctc-compress-strategy avg`.

### Decoder-only

The decoder-only models were obtained with the same script fo the decoder-prepending ones,
but setting the number of encoder layers to 0 and increasing the number of decoder layers.
This means that line 3 of Table 1 was obtained with:

```shell
python fbk-fairseq-dev/train.py $data_root \
	--train-subset $train_tsv --valid-subset $dev_tsv --config-yaml $config \
	--save-dir $save_dir --user-dir fbk-fairseq/examples/speech_to_text \
	--task speech_to_text_ctc --criterion label_smoothed_cross_entropy \
	--label-smoothing 0.1 \
	--arch s2tlm_transformer_fbk --encoder-layers 0 --decoder-layers 18 \
	--optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 25000 \
	--clip-norm 10.0 --adam-betas '(0.9, 0.98)' \
	--seed 1 --skip-invalid-size-inputs-valid-test \
	--update-freq 4 --max-tokens 40000 --num-workers 4 \
	--max-update 100000 --patience 10 --keep-last-epochs 12 \
	--log-format simple >> $save_dir/train.log 2> $save_dir/train.err
```

And causal masking can be enforced adding `--causal-prompt-mask`.

## Evaluation

We generate the hypothesis for our models with the command:

```shell
python fbk-fairseq/fairseq_cli/generate.py $DATA_ROOT \
        --user-dir fbk-fairseq/examples/speech_to_text/ --config-yaml $CONFIG_YAML \
        --gen-subset $SPLIT  \
        --max-tokens 80000 --unkpen 10000 --beam 5 \
        --max-source-positions 12000 --max-target-positions 4000 \
        --model-overrides "{'max_source_positions':12000,'max_target_positions':4000}" \
        --task speech_to_text_ctc --criterion label_smoothed_cross_entropy --no-repeat-ngram-size 5 \
        --path $MODEL
```

For models trained with the auxiliary CTC loss, change the `--criterion`
to `ctc_multi_loss` and add `--underlying-criterion label_smoothed_cross_entropy`.

### WER

WER was computed using jiWER after removing punctuation. This was done with the following script:

```shell
ref=$1
out=$2
tmp_dir=$(mktemp -d -t ci-XXXXXXXXXX)
tr -d '[:punct:]' < $ref | sed 's/  / /g' > $tmp_dir/ref.txt
tr -d '[:punct:]' < $out | sed 's/  / /g' > $tmp_dir/out.txt

jiwer -h $tmp_dir/out.txt -r $tmp_dir/ref.txt
rm -rf $tmp_dir
```

The statistical significance was computed using the script 
[WER bootstrap resampling](../examples/speech_to_text/scripts/wer_bootstrap_resampling.py).

### BLEU

All the scores and statistical significance were computed with the `sacreBLEU` command.

## Citation

```bibtex
@inproceedings{lam-et-al-2025-prepending,
  title={{Prepending or Cross-Attention for Speech-to-Text? An Empirical Comparison}},
  author={Tsz Kin Lam and Marco Gaido and Sara Papi and Luisa Bentivogli and Barry Haddow},
  booktitle = "Proceedings of the 2025 Annual Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics",
  address = "Albuquerque, New Mexico",
  year={2025}
}
```

