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

TBD
