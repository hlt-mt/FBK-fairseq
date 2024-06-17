# StreamAtt: Direct Streaming Speech-to-Text Translation with Attention-based Audio History Selection (ACL 2024)
![ACL Anthology](https://img.shields.io/badge/anthology-brightgreen?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8%2BCjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgo8c3ZnCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgdmVyc2lvbj0iMS4wIgogICB3aWR0aD0iNjgiCiAgIGhlaWdodD0iNjgiCiAgIGlkPSJzdmcyIj4KICA8ZGVmcwogICAgIGlkPSJkZWZzNCIgLz4KICA8cGF0aAogICAgIGQ9Ik0gNDEuOTc3NTUzLC0yLjg0MjE3MDllLTAxNCBDIDQxLjk3NzU1MywxLjc2MTc4IDQxLjk3NzU1MywxLjQ0MjExIDQxLjk3NzU1MywzLjAxNTggTCA3LjQ4NjkwNTQsMy4wMTU4IEwgMCwzLjAxNTggTCAwLDEwLjUwMDc5IEwgMCwzOC40Nzg2NyBMIDAsNDYgTCA3LjQ4NjkwNTQsNDYgTCA0OS41MDA4MDIsNDYgTCA1Ni45ODc3MDgsNDYgTCA2OCw0NiBMIDY4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDEwLjUwMDc5IEwgNTYuOTg3NzA4LDMuMDE1OCBDIDU2Ljk4NzcwOCwxLjQ0MjExIDU2Ljk4NzcwOCwxLjc2MTc4IDU2Ljk4NzcwOCwtMi44NDIxNzA5ZS0wMTQgTCA0MS45Nzc1NTMsLTIuODQyMTcwOWUtMDE0IHogTSAxNS4wMTAxNTUsMTcuOTg1NzggTCA0MS45Nzc1NTMsMTcuOTg1NzggTCA0MS45Nzc1NTMsMzAuOTkzNjggTCAxNS4wMTAxNTUsMzAuOTkzNjggTCAxNS4wMTAxNTUsMTcuOTg1NzggeiAiCiAgICAgc3R5bGU9ImZpbGw6I2VkMWMyNDtmaWxsLW9wYWNpdHk6MTtmaWxsLXJ1bGU6ZXZlbm9kZDtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MTIuODk1NDExNDk7c3Ryb2tlLWxpbmVjYXA6YnV0dDtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLW1pdGVybGltaXQ6NDtzdHJva2UtZGFzaGFycmF5Om5vbmU7c3Ryb2tlLWRhc2hvZmZzZXQ6MDtzdHJva2Utb3BhY2l0eToxIgogICAgIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsIDExKSIKICAgICBpZD0icmVjdDIxNzgiIC8%2BCjwvc3ZnPgo%3D&label=ACL&labelColor=white&color=red)

Code for the paper: ["StreamAtt: Direct Streaming Speech-to-Text Translation with Attention-based Audio History Selection"](https://arxiv.org/abs/2406.06097) published at the ACL 2024 main conference.

## 📎 Requirements
To run the agent, please make sure that [this repository](../README.md#installation) and 
[SimulEval v1.1.0](https://github.com/facebookresearch/SimulEval/commit/ec759d124307096dbbf6c3269d2ed652cc15fbdd) 
are installed.

Create a textual file (e.g., `src_audiopath_list.txt`) containing the list of paths to the audio 
files (one path per line for each file), which, differently from SimulST, are __not__ split into 
segments but are the entire speeches.
Specifically, in the case of the MuST-C dataset used in the paper, the file contains the paths to 
the entire TED talk files, similar to the following:
```txt
${AUDIO_DIR}/ted_1096.wav
${AUDIO_DIR}/ted_1102.wav
${AUDIO_DIR}/ted_1104.wav
${AUDIO_DIR}/ted_1114.wav
${AUDIO_DIR}/ted_1115.wav
...
```
Instead, as target file `translations.txt`, it can either be used a dummy file or the sentences 
concatenation, one line for each talk. 
However, for the evaluation of already segmented test sets, such as in MuST-C, we will not need 
these references, and we will evaluate directly from the segmented translations provided with the 
dataset, as described in [Evaluation with StreamLAAL](#-evaluation-streamlaal).

## 📌 Pre-trained Offline models
⚠️ The offline ST models used for the baseline, AlignAtt, and StreamAtt are the same and already available at 
the [AlignAtt release webpage](ALIGNATT_SIMULST_AGENT_INTERSPEECH2023.md#-pre-trained-offline-models)❗ 

## 🤖 Streaming Inference: *StreamAtt*
For the streaming inference, set `--config` and `--model-path` as, respectively, the config file 
and the model checkpoint downloaded in the 
[Pre-trained Offline models](#-pre-trained-offline-models) step.
As `--source` and `--target`, please use the files `src_audiopath_list.txt` and `translations.txt` 
created in the [Requirements](#-requirements) step.

The output will be saved in `--output`.

### ⭐ StreamAtt
For the ***Hypothesis Selection*** (based on AlignAtt), please set `--frame-num` as the value of 
*f* used for the inference (`f=[2, 4, 6, 8]`, in the paper).

Depending on the ***Textual History Selection*** ([Fixed Words](#fixed-words) or [Punctuation](#punctuation)), run the following command:

#### Fixed Words
```bash
simuleval \
    --agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.streaming_st_agent.StreamingSTAgent \
    --simulst-agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_offline_alignatt.AlignAttSTAgent \
    --history-selection-method examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.text_first_history_selection.FixedWordsHistorySelection \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --config config.yaml \
    --model-path checkpoint.pt \
    --source-segment-size 1000 \
    --extract-attn-from-layer 3 \
    --frame-num ${FRAME} \
    --history-words 20 \
    --quality-metrics BLEU --latency-metrics LAAL AL ATD --computation-aware --output ${OUT_DIR} \
    --device cuda:0
```

#### Punctuation
```bash
simuleval \
    --agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.streaming_st_agent.StreamingSTAgent \
    --simulst-agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_offline_alignatt.AlignAttSTAgent \
    --history-selection-method  examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.text_first_history_selection.PunctuationHistorySelection \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --config config.yaml \
    --model-path checkpoint.pt \
    --source-segment-size 1000 \
    --extract-attn-from-layer 3 \
    --frame-num ${FRAME} \
    --quality-metrics BLEU --latency-metrics LAAL AL ATD --computation-aware --output ${OUT_DIR} \
    --device cuda:0
```

### ⭐ Baseline and Upperbound

To run the baseline, execute the following command:
```bash
simuleval \
    --agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.streaming_st_agent.StreamingSTAgent \
    --simulst-agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_offline_alignatt.AlignAttSTAgent \
    --history-selection-method examples.speech_to_text.simultaneous_translation.agents.v1_1.streaming.text_first_history_selection.FixedAudioHistorySelection \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --config config.yaml \
    --model-path checkpoint.pt \
    --source-segment-size 1000 \
    --extract-attn-from-layer 3 \
    --frame-num ${FRAME} \
    --history-words 20 \
    --quality-metrics BLEU --latency-metrics LAAL AL ATD --computation-aware --output ${OUT_DIR} \
    --device cuda:0
```

For the simultaneous inference with AlignAtt (the upperbound presented in the paper), please refer 
to the [AlignAtt README](ALIGNATT_SIMULST_AGENT_INTERSPEECH2023.md#-inference).

## 💬 Evaluation: *StreamLAAL*
To evaluate the streaming outputs, download and extract the 
[mwerSegmenter](https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz) in the
`${MWERSEGMENTER_DIR}` folder, and run the following command: 
```bash
export MWERSEGMENTER_ROOT=${MWERSEGMENTER_DIR}

streamLAAL --simuleval-instances ${SIMULEVAL_INSTANCES}  \
           --reference ${REFERENCE_TEXTS} \
           --audio-yaml ${AUDIO_YAML} \
           --sacrebleu-tokenizer ${SACREBLEU_TOKENIZER} \
           --latency-unit ${LATENCY_UNIT}
```
where `${SIMULEVAL_INSTANCES}` is the output `instances.log` produced by the agent in the previous 
step, `${REFERENCE_TEXTS}` are the textual references in the target language (one line for each 
segment), `${AUDIO_YAML}` is the yaml file containing the original audio segmentation, 
`${SACREBLEU_TOKENIZER}` is the [sacreBLEU](https://github.com/mjpost/sacrebleu) tokenizer used for
the quality evaluation (defaults to `13a`), and `${LATENCY_UNIT}` is the unit used for the latency
computation (either `word` or `char`, defaults to `word`, the unit used in the paper).

If invoking `streamLAAL` does not work, please include the FBK-fairseq directory 
(`${FBK_FAIRSEQ_DIR}`) in the `PYTHONPATH` (`export PYTHONPATH=${FBK_FAIRSEQ_DIR}:$PYTHONPATH`) or 
call it explicitly by running 
`python ${FBK_FAIRSEQ_DIR}/examples/speech_to_text/simultaneous_translation/scripts/stream_laal.py`.


## 📍Citation
```bibtex
@inproceedings{papi-et-al-2024-streamatt,
title = {{StreamAtt: Direct Streaming Speech-to-Text Translation with Attention-based Audio History Selection}},
author = {Papi, Sara and Gaido, Marco and Negri, Matteo and Bentivogli, Luisa},
booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
year = {2024},
address = "Bangkok, Thailand",
}
```
