# SPES: Spectrogram Perturbation for Explainable Speech-to-Text Generation


This README contains instructions to generate feature attribution explanations of the outputs of 
speech-to-text models using SPES and evaluate them.
SPES is introduced in [SPES: Spectrogram Perturbation for Explainable Speech-to-Text Generation](https://arxiv.org/abs/2411.01710).

The explanations consist in saliency maps of relevance scores for each value in the spectrogram 
representations of the input audio, as well as relevance scores for the previously generated text tokens. 
To assign these relevance scores, SPES performs multiple forward-passes on the same data with different parts 
of the input masked. 
The impact of these occlusions is measured by comparing the original output probability distributions to that 
produced by the occluded inferences. The more the probability distribution changes, the more the masked 
part of the input is considered to be relevant.  

## 0. Preprocess the data

If the data is not preprocessed, follow the preprocessing steps of [Speechformer](https://gitlab.fbk.eu/mt/fbk-fairseq/-/blob/internal_master/fbk_works/SPEECHFORMER.md#preprocessing).

## 1. Perform a standard inference

The first step involves performing a standard inference and saving the model's predictions. These predicted tokens will be used to apply forced decoding when performing the occluded inferences. 

This can be done with the following script, where `data_dir` should be the directory where the tsv file containing the preprocessed data is stored, `tsv_file` the name of that file (without the .tsv extension), `model_path` the path to the fairseq model checkpoint to be used, `model_yaml_config` the model's configuration, and `output_file` the path where to store the standard output. `explanation_tsv` should be the name of the output file that will contain the tokens predicted by the model. 

In this and the following scripts, the argument `--max-tokens` should be adjusted based on the GPU's VRAM capacity.
```bash
python /fbk-fairseq/fairseq_cli/generate.py ${data_dir} \
	--gen-subset ${tsv_file} \
	--user-dir examples/speech_to_text \
	--max-tokens 40000 \
	--config-yaml ${model_yaml_config} \
	--beam 5 \
	--task speech_to_text_ctc \
	--criterion ctc_multi_loss \
	--underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
	--no-repeat-ngram-size 5 \
	--path ${model_path} > ${output_file}

# Saves the tokenized translation hypotheses to a tab separated file
python /fbk-fairseq/examples/speech_to_text/scripts/xai/prep_hyps_for_explanation.py \
  --model-output ${output_file} \
  --original-tsv ${data_dir}/${tsv_file}.tsv \
  --explain-tsv ${explanation_tsv}
```

## 2. Save output probabilities

The second step consists in running a standard inference and storing the output probability distributions. These stored distributions will serve as a reference for computing relevance scores. Relevance is determined by occluding different parts of the input, running 'perturbed' inferences, and comparing their probability distributions to the reference distributions. This comparison quantifies the extent to which each perturbation affects the model's output.

In the following script, `explain_tsv_file` should be the name of the file generated in the previous step (without the .tsv extension), `data_dir` the directory where it is stored, and `output_file` where to store the probabilities (without the .h5 extension).
In this step, the configuration file (`explain_yaml_config`) is identical to `model_yaml_config`, except that it omits the `bpe_tokenizer` field, as the target text in `explain_tsv_file` for forced decoding is already tokenized.
Other variables are the same as in the previous step.

```bash
python /fbk-fairseq/examples/speech_to_text/get_probs_from_constrained_decoding.py ${data_dir} \
    --gen-subset ${explain_tsv_file} \
    --user-dir examples/speech_to_text \
    --max-tokens 10000 \
    --config-yaml ${explain_yaml_config} \
    --task speech_to_text_ctc \
    --criterion ctc_multi_loss \
    --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --path ${model_path} \
    --save-file ${output_file}
```

## 3. Perform occluded inferences and generate the explanations

Next, multiple inferences are performed with different parts of the input occluded. The matrices of relevance computed from those are stored in a .h5 file.

`probs_path` should be the path to the file generated in the previous step, and `explanations_path` where to store the explanation heatmaps (without the .h5 extension). 

```bash
python /fbk-fairseq/examples/speech_to_text/generate_occlusion_explanation.py ${data_dir} \
    --gen-subset ${explain_tsv_file} \
    --user-dir examples/speech_to_text \
    --max-tokens 100000 \
    --num-workers 0 \
    --config-yaml ${explain_yaml_config} \
    --perturb-config ${occlusion_config} \
    --task speech_to_text_ctc \
    --criterion ctc_multi_loss \
    --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --no-repeat-ngram-size 5 \
    --path ${model_path} \
    --original-probs ${probs_path} \
    --save-file ${explanations_path}
```

#### Example of `occlusion_config.yaml`

`occlusion_config` should be a .yaml file containing the parameters with which to perform the occlusions. Below is an example how these files should be structured and a set of values that can be used. More information on the meaning of each of these parameters can be found in the [perturbator](https://gitlab.fbk.eu/mt/fbk-fairseq/-/tree/internal_master/examples/speech_to_text/occlusion_explanation/perturbators) package.

```
fbank_occlusion:
  category: slic_fbank_dynamic_segments
  p: 0.5
  n_segments: [2000, 2500, 3000]
  threshold_duration: 750
  n_masks: 20000
decoder_occlusion:
  category: discrete_embed
  p: 0.0
  no_position_occlusion: true
scorer: KL
```

## 4. Evaluate explanations

To evaluate the explanations, inference is performed with different percentages of the most relevant features occluded. This percentage is increased in `perc-interval` increments. 

In the following script, `tsv_file` should be the original preprocessed data. The translation hypotheses obtained with different levels of occlusion are stored in the `output_file`. All other arguments are the same as in previous steps.

In this step, the file passed as `model_yaml_config` should again contain a field `bpe_tokenizer` since we turn back to using beam search (as in step 1), rather than forced decoding (steps 2 and 3).

```bash
python /fbk-fairseq/fairseq_cli/generate.py ${data_dir} \
    --gen-subset ${tsv_file} \
    --user-dir examples/speech_to_text \
    --max-tokens 200000 \
    --config-yaml ${model_yaml_config} \
    --beam 5 \
    --max-source-positions 10000 \
    --max-target-positions 1000 \
    --task feature_attribution_evaluation_task \
    --aggregator sentence \
    --criterion ctc_multi_loss \
    --underlying-criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --no-repeat-ngram-size 5 \
    --explanation-path ${explanations_path} \
    --metric deletion \
    --normalizer single_mean_std paired_min_max \
    --perc-interval 5 \
    --path ${model_path} > ${output_file}
```

### Example Output File Structure

The `output_file` contains the hypotheses obtained with different levels of feature occlusion. Below is an example snippet of what the file might look like for sentence 739 with occlusions going from 0 to 20%. Lines starting with D contain the hypotheses output by the system.

```
T-739-2	he said fish
H-739-2	-0.7054908275604248	▁F ant as tic .
D-739-2	-0.7054908275604248	Fantastic.
P-739-2	-3.0391 -0.6950 -0.0423 -0.0290 -0.2239 -0.2036
T-739-1	he said fish
H-739-1	-0.2201979160308838	▁Some ▁fish .
D-739-1	-0.2201979160308838	Some fish.
P-739-1	-0.3025 -0.1083 -0.3261 -0.1439
T-739-0	he said fish
H-739-0	-0.25894302129745483	▁It ' s ▁a ▁fish .
D-739-0	-0.25894302129745483	It's a fish.
P-739-0	-0.7121 -0.2012 -0.1306 -0.4108 -0.0699 -0.1558 -0.1321
```

The script below can be used to calculate the AUC score, corresponding to the area below the curve when plotting the percentage of most relevant features occluded against the score of the output generated with that percentage of occlusion. The metric used for scoring the output is defined by the variable `scorer`, which can be `wer`, `wer_max` or `sacrebleu`. `output_file` should be the file generated by the script above, `reference_txt` a text file containing the reference sentences, and `figure_path` where to save the plot from which the AUC score is calculated.

```bash
python /fbk-fairseq/examples/speech_to_text/xai_metrics/auc_score.py \
    --reference ${reference_txt} \
    --output-path ${output_file} \
    --perc-step 5 \
    --scorer ${scorer} \
    --fig-path ${figure_path}
```

### Citation

```
@misc{fucci2024spesspectrogramperturbationexplainable,
      title={SPES: Spectrogram Perturbation for Explainable Speech-to-Text Generation}, 
      author={Dennis Fucci and Marco Gaido and Beatrice Savoldi and Matteo Negri and Mauro Cettolo and Luisa Bentivogli},
      year={2024},
      eprint={2411.01710},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.01710}, 
}```
