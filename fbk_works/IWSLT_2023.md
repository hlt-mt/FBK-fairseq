# Direct Models for Simultaneous Translation and Automatic Subtitling (IWSLT2023)
Models and inference scripts for the paper: [Direct Models for Simultaneous Translation and Automatic Subtitling: FBK@IWSLT2023](https://aclanthology.org/2023.iwslt-1.11/).

## 💬 Simultaneous Speech Translation

We release the offline ST model used for the FBK participation to the Simultaneous Speech Translation task: [**model folder**](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/EnnwDZFnXJdNjlhrKPqtNm8BHPz2d0E316Pp-yBy-dBpTg?e=Vhdvaw).

### 🤖 Inference with AlignAtt and EDAtt
Please install [SimulEval v1.1.0](https://github.com/facebookresearch/SimulEval/) (commit [3c19e1c](https://github.com/facebookresearch/SimulEval/commit/3c19e1c5e5deee043ab938d9b51996d5578b626c)) to run the evaluation.

#### 📌 AlignAtt
Set the parameters as described in [AlignAtt README](fbk_works/ALIGNATT_SIMULST_AGENT_INTERSPEECH2023.md) and 
run the following code:
```bash
simuleval \
    --agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_offline_alignatt.AlignAttSTAgent \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --config config_simul.yaml \
    --model-path ${ST_SAVE_DIR}/avg7.pt --prefix-size 1 --prefix-token "nomt" \
    --extract-attn-from-layer 3 --frame-num $FRAMES \
    --source-segment-size 1000 \
    --device cuda:0 \
    --quality-metrics BLEU --latency-metrics LAAL AL ATD --computation-aware \
    --output ${OUT_DIR}
```

#### 📌 EDAtt
Set the parameters as described in [EDAtt README](fbk_works/EDATT_SIMULST_AGENT_ACL2023.md) and 
run the following code:
```bash
simuleval \
    --agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_offline_edatt.EDAttSTAgent \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --config config_simul.yaml \
    --model-path ${ST_SAVE_DIR}/avg7.pt --prefix-size 1 --prefix-token "nomt" \
    --extract-attn-from-layer 3 --frame-num 2 --attn-threshold ${ALPHA} \
    --source-segment-size 1000 \
    --device cuda:0 \
    --quality-metrics BLEU --latency-metrics LAAL AL ATD --computation-aware \
    --output ${OUT_DIR}
```

## 📺 Automatic Subtitling

We release the Automatic Subtitling models for the FBK participation to the Automatic Subtitling task: 
- [**en-de model folder**](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/Es7feuTJ0phEqt450DN7clYBa_GdFfoZxpL5rBf-ix4ubQ?e=fxb01K) 
- [**en-es model folder**](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/Emn1YEgB2iBIq2LhMY4lNUcBnriFPTaUmHgWEXtJmM89xQ?e=UePzIQ)

For instructions of use, please refer to the [Direct Speech Translation for Automatic Subtitling README](fbk_works/DIRECT_SUBTITLING.md).

## 📍Citation
```bibtex
@inproceedings{papi-etal-2023-direct,
    title = "Direct Models for Simultaneous Translation and Automatic Subtitling: {FBK}@{IWSLT}2023",
    author = "Papi, Sara  and
      Gaido, Marco  and
      Negri, Matteo",
    booktitle = "Proceedings of the 20th International Conference on Spoken Language Translation (IWSLT 2023)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada (in-person and online)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.iwslt-1.11",
    doi = "10.18653/v1/2023.iwslt-1.11",
    pages = "159--168",
    }
```
