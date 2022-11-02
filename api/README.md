# External Interfaces

This package contains the interfaces that can be used by external
tools to access and use fairseq models.

## Simple IO Server

This server consists in a simple, single-thread server that serves a
speech translation model by accepting requests by reading one request
per line from the stdin (standard input) and answering the request
in the standard output.

### API

Each request in the stdin should be formulated with a JSON data structure
that **must** be put on a single, self-contained line.
Two types of requests are supported: commands, and inputs.
The only command supported so far is `shutdown` i.e., the server
can be shutdown by the client by sending the following request:

```json
{"command": "shutdown"}
```
The input requests instead contain three field: `wav_path`, `src_lang`, and `tgt_lang`.
An example of a valid request is:

```json
{"wav_path": "/this/is/the/path/to/a/wav/file.wav", "src_lang":  "en", "tgt_lang":  "it"}
```

Different processors may add other fields. Please refer to the
specific processor for more details. Currently, the processors are
`st`, and `st_triangle` that both use only those fields.
The server produces in the stdout (standard output) a different
response according to the processor used, but it is always formatted
as a single line JSON.
In the case of `st_triangle`,
the response fields are `score`,
`translation`, and `transcript` that respectively represent
_i)_ the model confidence on the generated output(s) expressed as log-probability,
_ii)_ the most likely translation, and
_iii)_ the most likely transcript, e.g.

```json
{"score": -2.4353, "translation":  "Il contenuto audio", "transcript":  "The audio content"}
```

### Usage

The server is meant to serve a specific model that is loaded at server
start time. To be started, the server accepts the same arguments of the
stardard fairseq generate, allowing to control all the aspects of the generation
in the same way. Namely, it requires a checkpoint of the model, the dictionaries,
a YAML configuration file, in addition to the generation arguments. An example of
the command to start the server is:

```bash
dir_with_dictionaries_and_yaml=$1
model_path=$2
conf_yaml=$3

python api/simple_io_server_st.py $data_bin \
    --user-dir examples/speech_to_text \
    --config-yaml $conf_yaml \
    --max-tokens 10000 \
    --scoring sacrebleu --beam 10 \
    --path $model_path \
    --max-source-positions 10000 --max-target-positions 1000 \
    --task speech_to_text_tagged_dual \
    --server-processor st_triangle
```

### Limitations

 - The server is single-thread and accepts only ONE request per time.
 - The server can serve only one model. To serve more models, start more servers.