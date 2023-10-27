# EDAtt agent for Simultaneous Speech Translation (ACL 2023)
Code for the paper: ["Attention as a Guide for Simultaneous Speech Translation"](https://arxiv.org/pdf/2212.07850.pdf) published at ACL 2023.

## 📎 Requirements
To run the agent, please make sure that [SimulEval v1.0.2](https://github.com/facebookresearch/SimulEval) (commit [d1a8b2f](https://github.com/facebookresearch/SimulEval/commit/d1a8b2f0b13fe5204f3dcb4935cae9c73dbfc285)) is installed 
and set `--port` accordingly.

## 📌 Pre-trained offline models
We release the offline ST models used for EDAtt simultaneous inference.
- **Common files**: [gcmvn.npz](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EVM-sFGlIitEnLrReN0P7kUBgayJ0rR5xB8dqvPYwjP8QQ?e=zdDoPJ) | [source_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/Eb5QB3M3NzNPnL_WTlPPKV8B3Def2irOo8v-8Y4gh9C0Rw?e=wsyzMr) | [source_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EXIWH7oKoUhKm-JfoOJF4YgBfAiwNRnmzdRxwbeve_Fo0g?e=qSK7f2)
- **en-de**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EUaIqYxA4pNKonM4JakqRWsBp4CZ-o-CqZzkH7d1W9AJtg?e=SqVIZ5) | [config_simul.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EZanfxsihW1Iv2nlegDZ0rwBbqWi0oCx_QfguPpd5tlhhg?e=bpZgtn) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/ESSS4UHoGYBKieVAallIxaABGfFarosDWTHtMpJgkGs7bA?e=5zXfMK) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/ETUZaRMsBd1Pu_Bv87XYe6EB3nJSS4lhO_xkftM-oC3kWA?e=Vb2FCF)
- **en-es**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EebJn5hHrLJGi1jfq0uwLQ0BYPgyICSzbZvroa6vf4JnXg?e=VdNU1o) | [config_simul.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EYrOY2ZpWb5Dq8XSdtStskQBMRAQnRQn41wxlVltSVkWcA?e=DqxUqo) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EXtxkcU9kYxAvIgCbprS-DMBPHP_OK1GprHfDoGYRLLGng?e=2rQ9fr) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/ERQXkC7Mb09KhhScQR0wS58BkgERv9_UwbXgrL-7RkHu_A?e=U1uW2N)

## 🤖 Inference
Set `--source`, `--target`, and `--config` as described in the 
[Fairseq Simultaneous Translation repository](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/simulst_mustc_example.md#inference--evaluation).
`--model-path` is the path to the offline ST model checkpoint (either en-de or en-es), 
`--attn-threshold` is the value of **alpha** used for the inference (`alpha=[0.6, 0.4, 0.2, 0.1, 0.05, 0.03]` in the paper).  
The output will be saved in `--output`.

```bash
simuleval \
    --agent ${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/agents/v1_0/simul_offline_edatt.py \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --config config_simul.yaml \
    --model-path ${ST_SAVE_DIR}/checkpoint_avg7.pt \
    --extract-attn-from-layer 3 \
    --frame-num 2 --attn-threshold $ALPHA \
    --speech-segment-factor 20 \
    --output ${OUT_DIR} \
    --port ${PORT} \
    --gpu \
    --scores
```

## 💬 Outputs
To ensure complete reproducibility, we also release the outputs obtained by EDAtt using SimulEval 1.0.2:
- **en-de**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/Ei5_n6-oJAJNoDhnfj1aQ9QBgtn2awGq6vN6dyyuRrMKlQ?e=nQGoAH)
- **en-es**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/ElCE1YTr4U9LlqpU8uBoXdoBeMQPpa7vIsjNGg5pIE1Rdg?e=tuxsHv)


## 📍Citation
```bibtex
@inproceedings{papi-et-al-2023-edatt,
title = "Attention as a Guide for Simultaneous Speech Translation",
author = {Papi, Sara and Negri, Matteo and Turchi, Marco},
booktitle = "Proceedings of the 61th Annual Meeting of the Association for Computational Linguistics",
year = "2023",
address = "Toronto, Canada",
publisher = "Association for Computational Linguistics"
}
```
