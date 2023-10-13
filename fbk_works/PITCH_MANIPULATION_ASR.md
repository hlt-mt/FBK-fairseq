# Pitch Manipulation to mitigate Gender Bias (ASRU 2023)

Code and models for the paper: 
["No Pitch Left Behind: Addressing Gender Unbalance in Automatic Speech Recognition Through Pitch Manipulation"](http://arxiv.org/abs/2310.06590) accepted at ASRU 2023.

## Models and Outputs

To ensure complete reproducibility, we release the ASR model checkpoints used in our experiments, 
together with the SentencePiece model, the vocabulary files, the yaml files, and the outputs obtained by each model:
- **Baseline**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EU7hsV5itRhNmCy5oMlbfDkBqOLDgg5dTuoGJijCviiFjg?e=FMY97r) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EZknSSZQBBVLhlKHWJ8mGyABTPs0z3dG7Vp4Ej3xKddc8Q?e=J1MsK9) | [tst-COMMON.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EeHgmf8Sq1pPnj_oqHD3sPgB2jv088b6xSbioig0FHuBHA?e=cs1ybz) | [tst-HE.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ER6Gnw9-c7hNonsiDD_R_fYBFdTZZ_Fjp4UNxlAfRhTWTg?e=gJoq4Y)
- **Baseline + VTLP**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EWxAZ9s63khJlICdcyiJfT8BidFvqLJNXx4Hg1lZNm_WLg?e=K6epon) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ETc2LnsLXixCl9nYnnMy6PAB9x4iZyjyijOhalyyvdE38w?e=LixlqS) | [tst-COMMON.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EezgNeYgcB5PrZbdqmf1POkBmHwgbcRLNyR4jTdVfB971A?e=dVlwYW) | [tst-HE.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EXP-4MzkN49KgtcgHzbTyE4BiPNhcT02dr3eqQvNoDZSBQ?e=qt16Re)
- **Baseline + Random**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EXG5GKwfnpVGu_po1Ruhok0BmzCvG9mBCIhFkHa5vWSIlQ?e=UliM5w) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EfXSf-bnRqtAov5CPXvKHiEBwDHoCmIthjk7MVzfmlX6qw?e=fye8mJ) | [tst-COMMON.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EcJerKWDNflIg514LHsfsPgB8l5lpR2LGas9ErLnZzfY4w?e=AiOdCK) | [tst-HE.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/Ef-K_h3mlE9Gi_arh39QT7cBwjGYXLdCTkmvNwu0KU7pUg?e=v68DR4)
- **Baseline + Opposite**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EZLMceYneNVLtNrnotVQvXsBLBrhDL7v1Pe7ZExmiAaWQA?e=8gqdgm) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EToxC9S2R-BEi4BCHCKzKxwBt-6G4WYkd8MC-MCI1Mpd6w?e=WAs43q) | [tst-COMMON.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ETYEECENPNhPk8Eu8aTHxHQB__us-1XGGN38e2z-qz7CZQ?e=ZivAcZ) | [tst-HE.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EaiuJorqnapDn3AaIJul_nYBImhjFDqIcVckiU64kKNbqA?e=SLcfE5)
- **Baseline + Random - Formant Shifting**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EdC5rqmQHrhEngGHcJADoOEBM4yGBArPKxHQT8g8OGiS-Q?e=mlhvBj) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/ERb5CET9agpNp8eN7aI3HzABqAKPuIWv8WRlZ9twSMACDA?e=cTHpzl) | [tst-COMMON.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EavObhcet_1GrMkJSRAjkswBek-7B9oBbhren90_mGA_TA?e=GmBSxT) | [tst-HE.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EY_Bs92CbLBKtXR0JqZ_jX0Buc4_lDQZ2xqCC2lQzmdW1A?e=iCsvTB)
- **Baseline + Random - Formant Shifting - Gender Swapping**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EekieFYtn1FBrU6QIdcBZdkB8_Pc0hdKfrGsweD1nDBqWg?e=gYBRg9) | [config.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EdxvFjxAsxRLg7ALmiRtLhYBaBacIwOZ2e81PvTES5mJuA?e=sFGy4h) | [tst-COMMON.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EavVRbKIaF5Hh7Uhv-xMNS0BdzsIbYcuITFUP2xJ9q9N1w?e=PpNvOW) | [tst-HE.out](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EeeTOA5wrd5BpN6om52VLVoBAHmYkEeStM56C30iBDBGOw?e=Hw3DFL)
- **Vocabulary**: [vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/dfucci_fbk_eu/Eb_aw_vKG1lDtKn8j9Xyy20BW6ImU8ODno9f-YlDk3RTEQ?e=8qecT7) | [spm_model](https://fbk-my.sharepoint.com/:u:/g/personal/dfucci_fbk_eu/EZLJOd-elFZEq8dLN3YlNCUBYefhJyhRJQt76zsZn6OTtA?e=yh0Kwa)


## Data Preprocessing

Data ([MuST-C v1](https://mt.fbk.eu/must-c/), en-es direction) have to be preprocessed with:

```bash
python /path/to/fbk-fairseq/examples/speech_to_text/preprocess_generic.py --data-root /data/to/mustc \
        --save-dir /data/to/mustc/save_folder --wav-dir /data/to/mustc/wav_folder \
        --split train, dev, tst-HE, tst-COMMON --vocab-type bpe --src-lang en --tgt-lang en \
        --task asr --n-mel-bins 80 --store-waveform
```

## Training

The following parameters are intended for training on a system with 4 GPUs, each having 16 GB of VRAM. 
The `training_data` and `dev_data` files are in TSV format, obtained after preprocessing.
The `config_file` is a YAML file and can be downloaded above.

```bash
python train.py /path/to/data_folder \
        --train-subset training_data --valid-subset dev_data \
        --save-dir /path/to/save_folder \
        --num-workers 5 --max-update 50000 --patience 10 --keep-last-epochs 13 \
        --max-tokens 10000 --adam-betas '(0.9, 0.98)' \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc --config-yaml config_file \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch conformer \
        --ctc-encoder-layer 8 --ctc-weight 0.5 \
        --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
        --warmup-updates 25000 \
        --clip-norm 10.0 \
        --seed 1 --update-freq 8 \
        --skip-invalid-size-inputs-valid-test \
        --log-format simple >> /path/to/save_folder/train.log 2> /path/to/save_folder/train.err


python /path/to/fbk-fairseq/scripts/average_checkpoints.py --input /path/to/save/folder  --num-epoch-checkpoints 5 --checkpoint-upper-bound $(ls /path/to/save_folder | head -n 5 | tail -n 1 | grep -o "[0-9]*") --output /path/to/save_folder/avg5.pt
```

## Inference

Inference can be executed with the following command
(setting `TEST_DATA` to a TSV obtained from the preprocessing
and `CONFIG_FILE` to one of the YAML files provided above):

```bash
python /path/to/fbk-fairseq/fairseq_cli/generate.py /path/to/data_folder \
        --gen-subset $TEST_DATA \
        --user-dir examples/speech_to_text \
        --max-tokens 40000 \
        --config-yaml $CONFIG_FILE \
        --beam 5 \
        --max-source-positions 10000 \
        --max-target-positions 1000 \
        --task speech_to_text_ctc \
        --criterion ctc_multi_loss \
        --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --no-repeat-ngram-size 5 \
        --path /path/to/checkpoint > /path/to/output_file
```

## Evaluation

We use the Python package [JiWER](https://pypi.org/project/jiwer/) to compute the word error rate.
Gender-specific evaluations are performed by partitioning the test sets based on the
[MuST-Speaker](https://mt.fbk.eu/must-speakers/) resource.


## Citation
```bibtex
@inproceedings{fucci2023pitch,
      title={{No Pitch Left Behind: Addressing Gender Unbalance in Automatic Speech Recognition through Pitch Manipulation}}, 
      author={Dennis Fucci and Marco Gaido and Matteo Negri and Mauro Cettolo and Luisa Bentivogli},
      year={2023},
      booktitle="IEEE Automatic Speech Recognition and Understanding Workshop (ASRU)",
      month = dec,
      address="Taipei, Taiwan"
}
```
