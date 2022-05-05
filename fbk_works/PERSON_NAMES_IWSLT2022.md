# Training

The models trained on multi-ligual source audio described in the paper have been created with the following scripts:

 - **Base**

```bash
# To be run on 4 GPUs (K80)
fairseq-train $data_root \
	--train-subset $train_sets --valid-subset $dev_set \
	--save-dir $st_save_dir \
	--num-workers 4 --max-update 100000 --patience 5 \
	--max-tokens 5000 \
	--user-dir examples/speech_to_text \
	--task speech_to_text --config-yaml $config_st \
	--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--arch s2t_transformer_m_fbk \
	--distance-penalty log \
	--optimizer adam \
	--lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 10000 \
	--clip-norm 10.0 \
	--seed 9 --update-freq 16 --load-pretrained-encoder-from $asr_pretrained \
	--skip-invalid-size-inputs-valid-test > ${st_save_dir}train.log 2> ${st_save_dir}train.err
```

 - **Triangle**

```bash
# To be run on 4 GPUs (K80)
fairseq-train $data_root \
	--train-subset $train_sets --valid-subset $dev_set \
	--save-dir $st_save_dir \
	--num-workers 2 --max-update 100000 --patience 5 \
	--max-tokens 5000 \
	--user-dir examples/speech_to_text \
	--task speech_to_text_ctc --config-yaml $config_st \
	--criterion cross_entropy_dualdecoder --label-smoothing 0.1 \
	--arch s2t_transformer_triangle_m \
	--distance-penalty log \
	--optimizer adam --find-unused-parameters \
	--lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 10000 \
	--clip-norm 10.0 \
	--seed 9 --update-freq 16 --load-pretrained-encoder-from $asr_pretrained \
	--skip-invalid-size-inputs-valid-test > ${st_save_dir}train.log 2> ${st_save_dir}train.err
```

 - **Triangle λ<sub>asr</sub> = 0.8, λ<sub>st</sub> = 0.2**

```bash
# To be run on 4 GPUs (K80)
fairseq-train $data_root \
	--train-subset $train_sets --valid-subset $dev_set \
	--save-dir $st_save_dir \
	--num-workers 4 --max-update 100000 --patience 5 \
	--max-tokens 5000 \
	--user-dir examples/speech_to_text \
	--task speech_to_text_ctc --config-yaml $config_st \
	--criterion cross_entropy_dualdecoder --label-smoothing 0.1 \
	--arch s2t_transformer_triangle_m \
	--distance-penalty log \
	--optimizer adam --find-unused-parameters \
	--lr 2e-3 --lr-scheduler inverse_sqrt \
	--warmup-updates 10000 \
	--clip-norm 10.0 \
	--auxiliary-loss-weight 0.8 --primary-loss-weight 0.2 \
	--seed 9 --update-freq 16 --load-pretrained-encoder-from $asr_pretrained \
	--skip-invalid-size-inputs-valid-test > ${st_save_dir}train.log 2> ${st_save_dir}train.err
```

# Inference

The output of the triangle models can be obtained using this script:

```bash
python examples/speech_to_text/generate_dualdecoder.py $data_bin \
    --user-dir examples/speech_to_text \
    --config-yaml $conf_yaml --gen-subset $split \
    --max-tokens 10000 --model-overrides "{'load_pretrained_encoder_from':None}" \
    --beam 10 \
    --path $model_path \
    --max-source-positions 10000 --max-target-positions 1000 \
    --task speech_translation_dualdecoding  > $out_path
```

while for the base model, it can be obtained running:

```bash
python fairseq_cli/generate.py $data_bin \
    --user-dir examples/speech_to_text \
    --config-yaml $conf_yaml --gen-subset $split \
    --max-tokens 10000 \
    --beam 5 \
    --path $model_path \
    --max-source-positions 10000 --max-target-positions 1000 \
    --task speech_to_text  > $out_path
```

# Models

We here release the pre-trained models that we used in our experiments.
The models correspond to those reported in Table 7 of the paper.
For each language pair we release the Sentencepiece dictionaries,
the Fairseq configuration files, and the checkpoints.


| dictionaries | config | base | triangle | triangle_08 |
|--------------|--------|------|----------|-------------|
|   [all-es](https://drive.google.com/file/d/1sli7_zFY-HbLuLay-odzWuGY6weEO0yD/view?usp=sharing)           |    [all-es](https://drive.google.com/file/d/1esJ6jffRV0iM5WNGGGlnOifBj2CJs0_W/view?usp=sharing)    |    [all-es](https://drive.google.com/file/d/1tlblxSQR01OvPaWJyYAa4QdlEKBksfVa/view?usp=sharing)  |      [all-es](https://drive.google.com/file/d/1bqkBnQ5d-K7Mp4LGhI3NPlbyYO6jEMEE/view?usp=sharing)    |      [all-es](https://drive.google.com/file/d/11-OlUYX2HV1NqkEUCZCfOE1QuL9qxKUH/view?usp=sharing)       |
|   [all-fr](https://drive.google.com/file/d/1-fLa4di9ucOBIzVHBVYWVIY3eM3IEIKe/view?usp=sharing)           |    [all-fr](https://drive.google.com/file/d/1qmMPTm-oxip81c9muyvwiChr9nQE3eIP/view?usp=sharing)    |    [all-fr](https://drive.google.com/file/d/1yI-IozrjILJmIpYFoozYYmmzAE0DGYbk/view?usp=sharing)  |      [all-fr](https://drive.google.com/file/d/1CGrQyAeLhLArqMTfumnSuZ9hzdYcRY8w/view?usp=sharing)    |      [all-fr](https://drive.google.com/file/d/1LJnUMwQhmChfWn-d0f_1QpcEGRK2sEQk/view?usp=sharing)       |
|   [all-it](https://drive.google.com/file/d/1ZSTpDvunl8m0L8ebMjdq4lmZwjaIVhKy/view?usp=sharing)           |    [all-it](https://drive.google.com/file/d/1wvt2cNMyAbtzJPnFqeljdodaaBFGxNaZ/view?usp=sharing)    |    [all-it](https://drive.google.com/file/d/1zO9nd39yqi-76nYLNe0RWxNXLktKbcC8/view?usp=sharing)  |      [all-it](https://drive.google.com/file/d/14HZa2RdPC5Pck2P9Uymrn7DBS4HbOZwI/view?usp=sharing)    |      [all-it](https://drive.google.com/file/d/1DsFHAMrWLly3oTGfvtPArKXd54BwAJYr/view?usp=sharing)       |


# Citation

```bibtex
@inproceedings{gaido-etal-2022-who,
    title = {{Who Are We Talking About? Handling Person Names in Speech Translation}},
    author = "Gaido, Marco  and Negri, Matteo  and Turchi, Marco",
    booktitle = "Proceedings of the 19th International Conference on Spoken Language Translation (IWSLT 2022)",
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics"
}
```
