# Training

Our Conformer model without pre-training was obtained running the following command
on 4 A40 GPUs (48GB RAM). With different hardware you may need to ajust the `--max-tokens` parameter
(in case your GPUs have less RAM) and the `--update-frequency` 
(to keep the product of `--max-tokens`, `--update-frequency`, and number of GPUs constant).


```bash
python train.py ${DATA_ROOT} \
        --train-subset ${COMMA_SEPARATED_TRAINING_SETS} \
	    --valid-subset dev_mustc2 \
        --save-dir ${ST_SAVE_DIR} \
	    --ignore-prefix-size 1 \
        --num-workers 5 --max-update 200000 --patience 30 --save-interval-updates 1000 \
        --max-tokens 40000 --adam-betas '(0.9, 0.98)' \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc --config-yaml config_st.yaml  \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch conformer \
        --ctc-encoder-layer 8 --ctc-weight 0.5 --ctc-compress-strategy avg \
        --optimizer adam --lr 2e-3 --lr-scheduler inverse_sqrt \
        --warmup-updates 25000 \
        --clip-norm 10.0 \
        --seed 1 --update-freq 2 \
        --skip-invalid-size-inputs-valid-test \
        --log-format simple > ${ST_SAVE_DIR}/train.log 2> ${ST_SAVE_DIR}/train.err
```

Similarly, the command to reproduce our fine-tuning is:

```bash
python train.py ${DATA_ROOT} \
        --train-subset ${COMMA_SEPARATED_TRAINING_SETS} \
	    --valid-subset dev_mustc2 \
        --save-dir ${ST_SAVE_DIR} \
	    --ignore-prefix-size 1 \
        --num-workers 5 --max-update 200000 --patience 30 --save-interval-updates 1000 \
        --max-tokens 40000 --adam-betas '(0.9, 0.98)' \
        --user-dir examples/speech_to_text \
        --task speech_to_text_ctc --config-yaml config_st.yaml  \
        --criterion ctc_multi_loss --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
        --arch conformer \
        --ctc-encoder-layer 8 --ctc-weight 0.5 --ctc-compress-strategy avg \
        --optimizer adam --lr 1e-3 --lr-scheduler fixed --reset-lr-scheduler --reset-optimizer --reset-dataloader \
        --clip-norm 10.0 \
        --seed 1 --update-freq 2 \
        --skip-invalid-size-inputs-valid-test \
        --log-format simple > ${ST_SAVE_DIR}/train.log 2> ${ST_SAVE_DIR}/train.err
```


# Models

## IWSLT Submission

We here release the trained models used in our participation to the competition.
They all share the same [configuration](https://drive.google.com/file/d/1xkmIxVs7qXgqqTAWXq0Cy1PCiGKL6gpx/view?usp=sharing),
[English](https://drive.google.com/file/d/125QgkB0YrcqKVBHwtizxZgt6FOkWPnt2/view?usp=sharing) and 
[German](https://drive.google.com/file/d/1zu0XFgRCUtCPz_okeModLizjW_QBZHF5/view?usp=sharing) dictionaries,
and [English](https://drive.google.com/file/d/1hB2djsdj2jj6-TXSScfnhw_F55m4XyZt/view?usp=sharing) and 
[German](https://drive.google.com/file/d/1bBKjG8KXJ3LfsHJrdAiAasF7V26otzmr/view?usp=sharing)
SentencePiece models.
The name of the models in this table are the same of Table 3 and 4 in the paper.
Please refer to the paper for a detailed description of each model.


| | Model | SacreBLEU MuST-C tst-COMMON | SacreBLEU MuST-C tst-COMMON SHAS-segmented |
|---|---|---|---|
| I. | [conformer](https://drive.google.com/file/d/1xreJg9fTb2Y0NRa1tHu0I9smzEpAsBX5/view?usp=sharing) | 30.6  | - |
| 1. | [conformer_indomainfn](https://drive.google.com/file/d/1QqwvAz07HCO2gvDBILfpthVWTi98XngY/view?usp=sharing) | 31.6  | 30.3 |
| 2. | [conformer_pretrain_indomainfn](https://drive.google.com/file/d/1Q8tlaBh5xiFEwGU7YGpuyB3K7c-OWq-p/view?usp=sharing) | 31.7 | 30.4 |
| 6. | [conformer_pretrain_indomainfn_resegmfn](https://drive.google.com/file/d/11A4nJxUuCe7VDH-trJN4uDp4XgIk50Gd/view?usp=sharing) | - | 29.7 |

## MuST-C only Models

We release here the best models mentioned in our paper trained only on MuST-C.
They share the same [config](https://drive.google.com/file/d/1ODBDkqdVjUKZmRivkT5_Iecc2KfYGzQj/view?usp=sharing)
file, [English](https://drive.google.com/file/d/19F1LLQfZsbv_KdBOCIo9zIPXixQBEdKM/view?usp=sharing)
and [German](https://drive.google.com/file/d/1GPhKDS9235tWFtNOJlFXXk2nlYtjqpqq/view?usp=sharing)
dictionaries, and 
[English](https://drive.google.com/file/d/1DdrERury5BNlGpWHE20KwTLNd6WQhGkl/view?usp=sharing)
and [German](https://drive.google.com/file/d/1C-y1zQ4DL1qhqgaQKiS3FK7fupygkpzq/view?usp=sharing)
Sentencepiece models.
Their description can be found in the paper, where  their results are reported in Table 1 and 2.
Namely, we release:

 - [conformer + CTC compr](https://drive.google.com/file/d/10sCbfFfgmkYKsV-kjMvyiZj8ZZYc-11v/view?usp=sharing) (25.5 BLEU): the best model without encoder pre-training;
 - [speechformer hybrid](https://drive.google.com/file/d/1RJPMmwg23tOL-H7ObPUACna603Vf0-yT/view?usp=sharing) (25.7 BLEU): the best model with encoder pre-training;
 - [conformer + CTC compr + char-ratio filter](https://drive.google.com/file/d/1pQAKYTCwi0dcBqWcD48TWKnSAj4k0dTK/view?usp=sharing) (26.7 BLEU): the best model obtained filtering the MuST-C training set; 

