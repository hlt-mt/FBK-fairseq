# Wait-k agent for offline ST systems

Agent for the paper: [Does Simultaneous Speech Translation need Simultaneous Models?](https://arxiv.org/abs/2204.03783)

To run the agent, please make sure that [SimulEval](https://github.com/facebookresearch/SimulEval) is installed and set `--port` accordingly. 

Set `--source`, `--target`, and `--config` as described in the [Fairseq Simultaneous Translation repository](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/simulst_mustc_example.md#inference--evaluation).
`--model-path` is the offline ST model checkpoint, 
`--lagging` is the value of **k_test** used for the wait-k inference (`lagging=[3, 5, 7, 9, 11]` in the paper).  
The simultaneous output will be saved in `--output`.

## Fixed Word Detection
```bash
simuleval \
    --agent ${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/agents/simul_offline_waitk.py \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --config ${CONFIG_YAML} --gpu \
    --model-path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --waitk ${LAGGING} \
    --speech-segment-factor 8 \
    --output ${OUT_DIR} \
    --port ${PORT} \
    --scores
```

## Adaptive Word Detection
```bash
simuleval \
    --agent ${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/agents/simul_offline_waitk.py \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --config ${CONFIG_YAML} --gpu \
    --model-path ${ST_SAVE_DIR}/${CHECKPOINT_FILENAME} \
    --waitk ${LAGGING} \
    --speech-segment-factor 8 \
    --adaptive-segmentation \
    --vocabulary-type sentencepiece \
    --output ${OUT_DIR} \
    --port ${PORT} \
    --scores
```

# Citation 
```bibtex
@inproceedings{papi-et-al-2022-does,
title = "Does Simultaneous Speech Translation need Simultaneous Models?",
author = {Papi, Sara and Gaido, Marco and Negri, Matteo and Turchi, Marco},
booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
year = "2022",
address = "Abu Dhabi, United Arab Emirates",
publisher = "Association for Computational Linguistics"
}
```
