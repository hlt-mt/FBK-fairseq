# SimulSeamless
![ACL Anthology](https://img.shields.io/badge/anthology-brightgreen?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiIHN0YW5kYWxvbmU9Im5vIj8%2BCjwhLS0gQ3JlYXRlZCB3aXRoIElua3NjYXBlIChodHRwOi8vd3d3Lmlua3NjYXBlLm9yZy8pIC0tPgo8c3ZnCiAgIHhtbG5zOnN2Zz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciCiAgIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyIKICAgdmVyc2lvbj0iMS4wIgogICB3aWR0aD0iNjgiCiAgIGhlaWdodD0iNjgiCiAgIGlkPSJzdmcyIj4KICA8ZGVmcwogICAgIGlkPSJkZWZzNCIgLz4KICA8cGF0aAogICAgIGQ9Ik0gNDEuOTc3NTUzLC0yLjg0MjE3MDllLTAxNCBDIDQxLjk3NzU1MywxLjc2MTc4IDQxLjk3NzU1MywxLjQ0MjExIDQxLjk3NzU1MywzLjAxNTggTCA3LjQ4NjkwNTQsMy4wMTU4IEwgMCwzLjAxNTggTCAwLDEwLjUwMDc5IEwgMCwzOC40Nzg2NyBMIDAsNDYgTCA3LjQ4NjkwNTQsNDYgTCA0OS41MDA4MDIsNDYgTCA1Ni45ODc3MDgsNDYgTCA2OCw0NiBMIDY4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDMwLjk5MzY4IEwgNTYuOTg3NzA4LDEwLjUwMDc5IEwgNTYuOTg3NzA4LDMuMDE1OCBDIDU2Ljk4NzcwOCwxLjQ0MjExIDU2Ljk4NzcwOCwxLjc2MTc4IDU2Ljk4NzcwOCwtMi44NDIxNzA5ZS0wMTQgTCA0MS45Nzc1NTMsLTIuODQyMTcwOWUtMDE0IHogTSAxNS4wMTAxNTUsMTcuOTg1NzggTCA0MS45Nzc1NTMsMTcuOTg1NzggTCA0MS45Nzc1NTMsMzAuOTkzNjggTCAxNS4wMTAxNTUsMzAuOTkzNjggTCAxNS4wMTAxNTUsMTcuOTg1NzggeiAiCiAgICAgc3R5bGU9ImZpbGw6I2VkMWMyNDtmaWxsLW9wYWNpdHk6MTtmaWxsLXJ1bGU6ZXZlbm9kZDtzdHJva2U6bm9uZTtzdHJva2Utd2lkdGg6MTIuODk1NDExNDk7c3Ryb2tlLWxpbmVjYXA6YnV0dDtzdHJva2UtbGluZWpvaW46bWl0ZXI7c3Ryb2tlLW1pdGVybGltaXQ6NDtzdHJva2UtZGFzaGFycmF5Om5vbmU7c3Ryb2tlLWRhc2hvZmZzZXQ6MDtzdHJva2Utb3BhY2l0eToxIgogICAgIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAsIDExKSIKICAgICBpZD0icmVjdDIxNzgiIC8%2BCjwvc3ZnPgo%3D&label=ACL&labelColor=white&color=red)

Code for the paper: ["SimulSeamless: FBK at IWSLT 2024 Simultaneous Speech Translation"](http://arxiv.org/abs/2406.14177) published at IWSLT 2024.

## 📎 Requirements
To run the agent, please make sure that 
[SimulEval v1.1.0](https://github.com/facebookresearch/SimulEval)
and [HuggingFace Transformers](https://huggingface.co/docs/transformers/index) are installed.

In the case of [💬 Inference using docker](#-inference-using-docker), use commit 
`f1f5b9a69a47496630aa43605f1bd46e5484a2f4` for SimulEval.

## 🤖 Inference using your environment
Please, set `--source`, and `--target` as described in the 
[Fairseq Simultaneous Translation repository](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/simulst_mustc_example.md#inference--evaluation): 
`${LIST_OF_AUDIO}` is the list of audio paths and `${TGT_FILE}` the segment-wise references in the 
target language. 

Set `${TGT_LANG}` as the target language code in 3 characters. The list of supported language 
codes is 
[available here](https://huggingface.co/facebook/hf-seamless-m4t-medium/blob/main/special_tokens_map.json).
For the source language, no language code has to be specified.

Depending on the target language, set `${LATENCY_UNIT}` to either `word` (e.g., for German) or 
`char` (e.g., for Japanese), and `${BLEU_TOKENIZER}` to either `13a` (i.e., the standard sacreBLEU 
tokenizer used, for example, to evaluate German) or `char` (e.g., to evaluate character-level 
languages such as Chinese or Japanese). 

The simultaneous inference of SimulSeamless is based on 
[AlignAtt](ALIGNATT_SIMULST_AGENT_INTERSPEECH2023.md), thus the __f__ parameter (`${FRAME}`) and the
layer from which to extract the attention scores (`${LAYER}`) have to be set accordingly. 

### Instruction to replicate IWSLT 2024 results ↙️

To replicate the results obtained to achieve 2 seconds of latency (measured by AL) on the test sets
used by [the IWSLT 2024 Simultaneous track](https://iwslt.org/2024/simultaneous), use the following
values:
- **en-de**: `${TGT_LANG}=deu`, `${FRAME}=6`, `${LAYER}=3`, `${SEG_SIZE}=1000`
- **en-ja**: `${TGT_LANG}=jpn`, `${FRAME}=1`, `${LAYER}=0`, `${SEG_SIZE}=400`
- **en-zh**: `${TGT_LANG}=cmn`, `${FRAME}=1`, `${LAYER}=3`, `${SEG_SIZE}=800`
- **cs-en**: `${TGT_LANG}=eng`, `${FRAME}=9`, `${LAYER}=3`, `${SEG_SIZE}=1000`

❗️Please notice that `${FRAME}` can be adjusted to achieve lower/higher latency.


The SimulSeamless can be run with:
```bash
simuleval \
    --agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_alignatt_seamlessm4t.AlignAttSeamlessS2T \
    --source ${LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --model-size medium --target-language ${TGT_LANG} \
    --extract-attn-from-layer ${LAYER} --num-beams 5 \
    --frame-num ${FRAME} \
    --source-segment-size ${SEG_SIZE} \
    --quality-metrics BLEU --latency-metrics LAAL AL ATD --computation-aware \
    --eval-latency-unit ${LATENCY_UNIT} --sacrebleu-tokenizer ${BLEU_TOKENIZER} \
    --output ${OUT_DIR} \
    --device cuda:0 
```
If not already stored in your system, the SeamlessM4T model will be downloaded automatically when 
running the script. The output will be saved in `${OUT_DIR}`. 

We suggest to run the inference using a GPU to speed up the process but the system can be run on 
any device (e.g., CPU) supported by SimulEval and HuggingFace.

## 💬 Inference using docker
To run SimulSeamless using docker, as required by the IWSLT 2024 Simultaneous track, follow the
steps below:
1. Download the docker file  [simulseamless.tar](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EWcMkUFCB59PtmtncHUmkRABGw-AwJn5iJ5Q8zIihfvnag?e=k6DxM0)
2. Load the docker image:
```bash
docker load -i simulseamless.tar
```
3. Start the SimulEval standalone with GPU enabled: 
```bash
docker run -e TGTLANG=${TGT_LANG} -e FRAME=${FRAME} -e LAYER=${LAYER} \
    -e BLEU_TOKENIZER=${BLEU_TOKENIZER} -e LATENCY_UNIT=${LATENCY_UNIT} \
    -e DEV=cuda:0 --gpus all --shm-size 32G \
    -p 2024:2024 simulseamless:latest
```
4. Start the remote evaluation with:
```bash
simuleval \
    --remote-eval --remote-port 2024 \
    --source ${LIST_OF_AUDIO} --target ${TGT_FILE} \
    --source-type speech --target-type text \
    --source-segment-size ${SEG_SIZE} \
    --eval-latency-unit ${LATENCY_UNIT} --sacrebleu-tokenizer ${BLEU_TOKENIZER} \
    --output ${OUT_DIR}
```
To set, `${TGT_LANG}`, `${FRAME}`, `${LAYER}`, `${BLEU_TOKENIZER}`, `${LATENCY_UNIT}`, 
`${LIST_OF_AUDIO}`, `${TGT_FILE}`, `${SEG_SIZE}`, and `${OUT_DIR}` refer to 
[🤖 Inference using your environment](#-inference-using-your-environment).

### Instruction to recreate the docker images <img height="20" width="25" src="https://cdn.jsdelivr.net/npm/simple-icons@v11/icons/docker.svg" />

To recreate the docker images, follow the steps below.

1. Download SimulEval and this repository. 
2. Create a `Dockerfile` with the following content:
```
FROM python:3.9
RUN pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
ADD /SimulEval /SimulEval
WORKDIR /SimulEval
RUN pip install -e .
WORKDIR ../
ADD /fbk-fairseq /fbk-fairseq
WORKDIR /fbk-fairseq
RUN pip install -e .
RUN pip install -r speech_requirements.txt
WORKDIR ../
RUN pip install sentencepiece
RUN pip install transformers

ENTRYPOINT simuleval --standalone --remote-port 2024 \
        --agent-class examples.speech_to_text.simultaneous_translation.agents.v1_1.simul_alignatt_seamlessm4t.AlignAttSeamlessS2T \
        --model-size medium --num-beams 5 --user-dir fbk-fairseq/examples \
        --target-language $TGTLANG --frame-num $FRAME --extract-attn-from-layer $LAYER --device $DEV \
        --sacrebleu-tokenizer ${BLEU_TOKENIZER} --eval-latency-unit ${LATENCY_UNIT}
```
3. Build the docker image:
```
docker build -t simulseamless .
```
4. Save the docker image:
```
docker save -o simulseamless.tar simulseamless:latest
```

## 📍Citation
```bibtex
@inproceedings{papi-et-al-2024-simulseamless,
title = "SimulSeamless: FBK at IWSLT 2024 Simultaneous Speech Translation",
author = {Papi, Sara and Gaido, Marco and Negri, Matteo and Bentivogli, Luisa},
booktitle = "Proceedings of the 21th International Conference on Spoken Language Translation (IWSLT)",
year = "2024",
address = "Bangkok, Thailand",
}
```
