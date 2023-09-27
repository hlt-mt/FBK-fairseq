# AlignAtt agent for Simultaneous Speech Translation (Interspeech 2023)
Code for the paper: ["AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation"](https://www.isca-speech.org/archive/pdfs/interspeech_2023/papi23_interspeech.pdf) published at Interspeech 2023.

## 📎 Requirements
To run the agent, please make sure that [SimulEval v1.0.2](https://github.com/facebookresearch/SimulEval) (commit [d1a8b2f](https://github.com/facebookresearch/SimulEval/commit/d1a8b2f0b13fe5204f3dcb4935cae9c73dbfc285)) is installed 
and set `--port` accordingly.

## 📌 Pre-trained offline models
We release the offline ST models used for AlignAtt simultaneous inference:
- **Common files**: [config_simul.yaml](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/Ec6V_VD--ApDq_N_qNaSzMgBqzsPtjRgw4KXphN4qmfDdA?e=X5S8Fg) | [gcmvn.npz](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EVM-sFGlIitEnLrReN0P7kUBgayJ0rR5xB8dqvPYwjP8QQ?e=zdDoPJ) | [source_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/Eb5QB3M3NzNPnL_WTlPPKV8B3Def2irOo8v-8Y4gh9C0Rw?e=wsyzMr) | [source_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EXIWH7oKoUhKm-JfoOJF4YgBfAiwNRnmzdRxwbeve_Fo0g?e=egZsxm)
- **en-de**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/ERN7FQPbgz9EqA7yx5PnNlUB-qsj6vRVVuvwrj7QWGv8eA?e=BaKqXp) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EalfrTxZUntBjFZDZDdoAqIBx7Bl5ODQ-lbZa6eIU3MUEw?e=HvxSUq) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EcvmsOCscihApu43f5w7P4UBHDBweGIve696G3Ip1MstAA?e=Opkd88)
- **en-es**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EQDfjdiau8NGo9RjM3RThVIBtX2aSJtzR6QnMVXyXYlB4Q?e=gt54tA) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EflcO5Xh7cdDuCCKS69bqL4BUEB9AJ68JbSQ4uehBl9WTQ?e=Gnn41C) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EaN1Ex8oX6BLpCzyeBjyWvABOHJj_loVn8cBv-27PsVMuQ?e=ukrlwd)
- **en-fr**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EeFePDdaD6xFu3n5_qF0ctkBWyUgI12R7xxvz3CmD8aMGg?e=dRcY2G) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EWG7G2--Gn9Jsj4yDBtfnU8BZAj_oqXHZFmJbVY9_nmBVQ?e=guBSOj) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/Ead49VOobUZLhO-Q9oy8qEkBGHp4OHtajuQwHfyWM9CZvQ?e=PGbKCv)
- **en-it**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EQMN21JadJBJtUYOVLvxqJoBwrg9_G9QOd26oZI5UZhNPA?e=cFCjC2) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EdkXa_WHAwdBitobYAcSpicBDRRw752opIhXS3fDzKrGfA?e=Kz8exm) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/ETBVj1iW__pKt1I45myEfRQB1c9en_PSGlVUdjaeJtYBrQ?e=bVX0cs)
- **en-nl**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EdeHs9WupaNAkH2tO6hVF9MBhnCs4AGbFj57d9E7WE2qDg?e=H4ZL6d) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EXDN_uHlhbRPiCXjKTEDoQsBpKJxO7Eq7JoGPS7oGbjmjw?e=lderWt) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/Ea5tH0sTJchIj4uTsTAN74gBNTAIby5KcEWnXkyroJ5IJQ?e=rpB7yz)
- **en-pt**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EUq0TG0z06FOhnYeBRijFYkBmbViXFyobZdxWE_5XZ0HbQ?e=v6lQrx) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/Ed_F15MAU9RLmzZjnXyhSd0BUGZUmdM9SYIJXr2vJFx4hw?e=A6ZkHD) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EahcXlA1QdZJvNSf0h0WcT8BzoXGXV7g3_1CB4OTCdRdUA?e=k8iWBD)
- **en-ro**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/ESTYU-VZVaBEgTIrIaetL5oB-fyKhjBys1ZP-NQY71vpYQ?e=Q8kIcq) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/Ec7F-LDB8JZMrgcDaHaPLLUBBhVW3Kchs_smpgKkdXgpbA?e=kRrzkg) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/EagKc8ELpvtOuaWpKt41gC4BSlzX0I3ql85_3QAiWW5JOw?e=b0R42Y)
- **en-ru**: [checkpoint](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EcbqPbOcjapEi0ynC6nzobQB0aBDOuIBDGjXp_4O1pl4eA?e=7im8jJ) | [target_vocab.model](https://fbk-my.sharepoint.com/:u:/g/personal/spapi_fbk_eu/EXs7Ehftq-5JomvtvoBc59cBCYNIJ5jqyg2MYR3YbaAuKg?e=k8J4fE) | [target_vocab.txt](https://fbk-my.sharepoint.com/:t:/g/personal/spapi_fbk_eu/ETwqlSVMjydHg_jPCW5Cn1gBjk7WPwoXvDQXeuoyQH0O_w?e=cr4AdP)

Please replace `spm_unigram8000_st_target.{model/txt}`, `spm_unigram.en.{model/txt}`, and `gcmvn.npz` of the `config_simul.yaml` with your absolute path to files.

## 🤖 Inference
For the simultaneous inference, set `--source`, `--target`, and `--config` as described in the 
[Fairseq Simultaneous Translation repository](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/simulst_mustc_example.md#inference--evaluation).
`--model-path` is the path to the offline ST model checkpoint, 
`--frame-num` is the value of **f** used for the inference (`f=[2, 4, 6, 8, 10, 12, 14]` in the paper).  
The output will be saved in `--output`.

```bash
simuleval \
    --agent ${FBK_FAIRSEQ_ROOT}/examples/speech_to_text/simultaneous_translation/agents/simul_offline_alignatt.py \
    --source ${SRC_LIST_OF_AUDIO} \
    --target ${TGT_FILE} \
    --data-bin ${DATA_ROOT} \
    --config config_simul.yaml \
    --model-path ${ST_SAVE_DIR}/checkpoint_avg7.pt \
    --extract-attn-from-layer 3 \
    --frame-num ${FRAME} \
    --speech-segment-factor 10 \
    --output ${OUT_DIR} \
    --port ${PORT} \
    --gpu \
    --scores
```
For the offline inference, please refer to the [Speechformer README](SPEECHFORMER.md#generate).

## 💬 Outputs
To ensure complete reproducibility, we also release the outputs obtained by AlignAtt using SimulEval 1.0.2:
- **en-de**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/Ev-Dkc0stNVMo0tCg3rwR0YBurTqZJOjh6FGTKICYgjwQA?e=N5Rwbz)
- **en-es**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/EmTf9nCL9nVDmsNAoRHcARkByaY667u4wopiHXcvy2dZyQ?e=fquhK8)
- **en-fr**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/ErsUFR-hHU5Omy3zCdaObOAB9g-v-EcdiClSmIFbOPk0lg?e=LVbmCg)
- **en-it**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/ErjR4sSFXudDv0gZdNaCJBcBChL_SEORM9y7vf0D-5KCNw?e=QSxeDY)
- **en-nl**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/Ei2gTDx5go5FlYaOTOt7tUUBmEFjRujpHq6KdSJD_B_Saw?e=00cQnu)
- **en-pt**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/EsHeu38ilU5EmN_00UJqheYBh8Hq31sjytsFZtxblU5hvA?e=z9u7UP)
- **en-ro**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/Ek9ynavxxlhLvrNlAOrLp9cBWiDBGd0AmEXF5DyVMZlZdQ?e=5Gu2Gz)
- **en-ru**: [outputs folder](https://fbk-my.sharepoint.com/:f:/g/personal/spapi_fbk_eu/Epftl62OIE1PoBzdHRMTZtMBZZZsjEcYbWarfs207EpTOg?e=G92Mq9)


## 📍Citation
```bibtex
@inproceedings{papi-et-al-2023-alignatt,
title = "AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation",
author = {Papi, Sara and Negri, Matteo and Turchi, Marco},
booktitle = "Proc. of Interspeech 2023",
year = "2023",
address = "Dublin, Ireland",
}
```
