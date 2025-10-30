# The Unheard Alternative: Contrastive Explanations for Speech-to-Text Models

Instructions to obtain and evaluate the contrastive feature attribution explanations for the generation of the gender terms in the [MuST-SHE dataset](https://aclanthology.org/2020.acl-main.619/), as in the paper "[The Unheard Alternative: Contrastive Explanations for Speech-to-Text Models](https://arxiv.org/abs/2509.26543)" published at BlackBoxNLP 2025.

### N.B. On the chosen ST model

The instructions below correspond to the [Transformer model used in the main body of our paper](https://github.com/facebookresearch/fairseq/blob/main/examples/speech_to_text/docs/mustc_example.md). If one of the [Conformer models for which we show results in the appendix](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/BUGFREE_CONFORMER.md) is used instead, the scripts `generate.py`, `get_probs_from_constrained_decoding.py`, and `generate_occlusion_explanation.py` should be run with the arguments `--criterion ctc_multi_loss`, `--underlying-criterion label_smoothed_cross_entropy`, and `--label-smoothing 0.1` added, and the `--prefix-size` omitted. The tasks `speech_to_text`,
`speech_to_text_genderxai`, and `feature_attribution_evaluation_task` should be replaced with `speech_to_text_ctc`,
`speech_to_text_genderxai_ctc`, and `feature_attribution_evaluation_task_ctc`, respectively.

## 0. Preprocess the data

If the data is not preprocessed, follow the preprocessing steps of [Speechformer](https://gitlab.fbk.eu/mt/fbk-fairseq/-/blob/internal_master/fbk_works/SPEECHFORMER.md#preprocessing).

## 1. Perform a standard inference

The first step involves performing a standard inference and saving the model's translation hypotheses. These predicted tokens will be used to apply forced decoding when performing the occluded inferences. 

This can be done with the following script, where `$data_dir` should be the directory where the tsv file containing the preprocessed data for MuST-SHE is stored, `$tsv_file` the name of that file (without the .tsv extension), `$model_path` the path to the fairseq model checkpoint to be used, `$model_yaml_config` the model's configuration, and `$output_file` the path where to store the standard output. `$explanation_tsv` should be the name of the output file that will contain the tokens predicted by the model. 

In this and the following scripts, the argument `--max-tokens` should be adjusted based on the GPU's VRAM capacity. The value indicated here are meant for a 40GB A40 GPU.

```bash
python /fbk-fairseq/fairseq_cli/generate.py ${data_dir} \
	--gen-subset ${tsv_file} \
	--user-dir examples/speech_to_text \
	--max-tokens 40000 \
	--model-overrides "{'batch_unsafe_relative_shift':False, 'load_pretrained_encoder_from':None}" \
	--max-source-positions 7000 \
	--config-yaml ${model_yaml_config} \
	--beam 5 \
	--task speech_to_text \
        --prefix-size 1 \
	--no-repeat-ngram-size 5 \
	--path ${model_path} > ${output_file}
```

After the generation of the outputs, save the tokenized translation hypotheses to a tab separated file with the following command:

```bash
python /fbk-fairseq/examples/speech_to_text/scripts/xai/prep_hyps_for_explanation.py \
  --model-output ${output_file} \
  --original-tsv ${data_dir}/${tsv_file}.tsv \
  --explain-tsv ${explanation_tsv}
```

Then, add relevant information for gender experiments on MuST-SHE, including the word pairs to contrast, to the tsv:

```bash
python /home/lvarellaconti/fbk-fairseq/examples/speech_to_text/scripts/xai/prep_mustshe_hyps_for_explanation.py \
	--hypotheses-tsv ${explanation_tsv} \
	--mustshe-tsv ${mustshe_tsv} \
	--spm-model ${spm_model} \
	--output-tsv ${gender_explanation_tsv}
```

## 2. Save output probabilities

The second step consists in storing the output probability distributions. These stored distributions will serve as a reference for computing relevance scores.

In the following script, `$gender_explanation_tsv` should be the name of the file generated in the previous step (without the .tsv extension), `$data_dir` the directory where it is stored, and `$output_file` where to store the probabilities (without the .h5 extension).
In this step, the configuration file (`$explain_yaml_config`) is identical to `$model_yaml_config`, except that it omits the `$bpe_tokenizer` field, as the target text in `$gender_explanation_tsv` for forced decoding is already tokenized.
Other variables are the same as in the previous step.

```bash
python /fbk-fairseq/examples/speech_to_text/get_probs_from_constrained_decoding.py ${data_dir} \
    --gen-subset ${gender_explanation_tsv} \
    --user-dir examples/speech_to_text \
    --max-tokens 10000 \
    --model-overrides "{'batch_unsafe_relative_shift':False, 'load_pretrained_encoder_from':None}" \
    --max-source-positions 7000 \
    --config-yaml ${explain_yaml_config} \
    --task speech_to_text_genderxai \
    --explanation-task gender \
    --prefix-size 1 \
    --path ${model_path} \
    --save-file ${output_file}
```

## 3. Perform occluded inferences and generate the explanations

Next, multiple inferences are performed with different parts of the input occluded. The matrices of relevance computed from those are stored in a .h5 file.

`$probs_path` should be the path to the file generated in the previous step, and `$explanations_path` where to store the explanation heatmaps (without the .h5 extension). 

```bash
python /fbk-fairseq/examples/speech_to_text/generate_occlusion_explanation.py ${data_dir} \
    --gen-subset ${gender_explanation_tsv} \
    --user-dir examples/speech_to_text \
    --max-tokens 100000 \
    --num-workers 0 \
    --model-overrides "{'batch_unsafe_relative_shift':False, 'load_pretrained_encoder_from':None}" \
    --max-source-positions 7000 \
    --config-yaml ${explain_yaml_config} \
    --perturb-config ${occlusion_config} \
    --task speech_to_text_genderxai \
    --prefix-size 1 \
    --path ${model_path} \
    --original-probs ${probs_path} \
    --save-file ${explanations_path}
```

#### Example of `occlusion_config.yaml`

`$occlusion_config` should be a YAML file containing the parameters with which to perform the occlusions. Below is an example how these files should be structured and a set of values that can be used. More information on the meaning of each of these parameters can be found in the [perturbator](https://gitlab.fbk.eu/mt/fbk-fairseq/-/tree/internal_master/examples/speech_to_text/occlusion_explanation/perturbators) and [scorers](https://gitlab.fbk.eu/mt/fbk-fairseq/-/tree/internal_master/examples/speech_to_text/occlusion_explanation/scorers) packages.

```
fbank_occlusion:
  category: slic_fbank_dynamic_segments
  p: 0.5
  n_segments: [2000, 2500, 3000]
  threshold_duration: 750
  n_masks: 20000
  slic_sigma: 0
decoder_occlusion:
  category: discrete_embed
  p: 0.0
  no_position_occlusion: true
scorer: 
  category: gender_term_contrastive_parity_ratio
  prob_aggregation: word_boundary
explanation_task: gender
```

## 4. Evaluate explanations

To evaluate the explanations, inference is performed with different percentages of the most relevant features occluded. This percentage is increased in `perc-interval` increments. 

In the following script, `$tsv_file` should be the original preprocessed data. The translation hypotheses obtained with different levels of occlusion are stored in the `$output_file`. All other arguments are the same as in previous steps.

In this step, the file passed as `$model_yaml_config` should again contain a field `$bpe_tokenizer` since we turn back to using beam search (as in step 1), rather than forced decoding (steps 2 and 3).

```bash
python /fbk-fairseq/fairseq_cli/generate.py ${data_dir} \
    --gen-subset ${gender_explanation_tsv} \
    --user-dir examples/speech_to_text \
    --max-tokens 200000 \
    --model-overrides "{'batch_unsafe_relative_shift':False, 'load_pretrained_encoder_from':None}" \
    --max-source-positions 7000 \
    --config-yaml ${model_yaml_config} \
    --beam 5 \
    --max-source-positions 10000 \
    --max-target-positions 1000 \
    --task feature_attribution_evaluation_task \
    --aggregator sentence \
    --prefix-size 1 \
    --no-repeat-ngram-size 5 \
    --explanation-path ${explanations_path} \
    --metric deletion \
    --perc-interval 1 \
    --max-percent 20 \
    --path ${model_path} > ${output_file}
```

The script below can be used to compute the gender coverage or flip rate at each deletion step. The `--scorer` should be either `gender_coverage` or `gender_flip_rate`. `$output_file` should be the file generated by the script above, `$mustshe_ext_tsv`, the path to the tsv file of the MuST-SHE extension containing part-of-speech annotations, and `$res_path`, the path to the numpy file were the score for each deletion step will be saved.

```bash
python /home/lvarellaconti/fbk-fairseq/examples/speech_to_text/xai_metrics/auc_probability_score_mustshe.py \
    --tsv-path ${gender_explanation_tsv} \
    --output-path ${output_file} \
    --perc-step 1 \
    --max-percent 20 \
    --scorer gender_coverage \
    --categories 1F 1M \
    --no-articles \
    --mustshe-pos ${mustshe_ext_tsv} \
    --data-path ${res_path}
```


## Citation

```
@misc{conti2025unheardalternativecontrastiveexplanations,
      title={The Unheard Alternative: Contrastive Explanations for Speech-to-Text Models}, 
      author={Lina Conti and Dennis Fucci and Marco Gaido and Matteo Negri and Guillaume Wisniewski and Luisa Bentivogli},
      year={2025},
      eprint={2509.26543},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26543}, 
}
```
