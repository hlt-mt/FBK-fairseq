# Copyright 2024 FBK

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

import argparse
import json
import os
import shutil
import subprocess
import tempfile
import yaml

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple

from simuleval.evaluator.scorers.latency_scorer import LAALScorer
from simuleval.evaluator.scorers.quality_scorer import SacreBLEUScorer


VERSION = "1.0.0"
_CITATION = r"""@inproceedings{papi-etal-2024-streamatt,
      title={{Direct Streaming Speech-to-Text Translation with Attention-based Audio History Selection}}, 
      author={Sara Papi and Marco Gaido and Matteo Negri and Luisa Bentivogli},
      booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
      address = "Bangkok, Thailand",
      year={2024}
}"""


@dataclass
class ReferenceSentenceDefinition:
    """
    Stores the information about the reference sentences.
    """
    content: str
    start_time: float
    duration: float


class SimulEvalLogInstance:
    """
    Stores the information about the output instances generated by SimulEval.
    """
    def __init__(self, info: Dict[str, Any], latency_unit: str = "word"):
        for key, value in info.items():
            setattr(self, key, value)

        self.reference = info.get("reference", "")
        self.prediction = info.get("prediction", "")
        self.latency_unit = latency_unit
        self.metrics = {}

    @property
    def reference_length(self) -> int:
        return self.string_to_len(self.reference, self.latency_unit)

    @staticmethod
    def string_to_len(string: str, latency_unit: str) -> int:
        if latency_unit == "word":
            return len(string.split(" "))
        elif latency_unit == "char":
            return len(string.strip())
        else:
            raise NotImplementedError


class MwerSegmenter:
    """
    Executes the mWERSegmenter tool introduced in `"Evaluating Machine Translation Output
    with Automatic Sentence Segmentation" by Matusov et al. (2005)
    <https://aclanthology.org/2005.iwslt-1.19/>`_.

    The tool can be downloaded at:
    https://www-i6.informatik.rwth-aachen.de/web/Software/mwerSegmenter.tar.gz
    """
    def __init__(self):
        self.mwer_command = "mwerSegmenter"
        if shutil.which(self.mwer_command) is None:
            mwerSegmenter_root = os.getenv("MWERSEGMENTER_ROOT")
            assert mwerSegmenter_root is not None, \
                f"{self.mwer_command} is not in PATH and no MWERSEGMENTER_ROOT environment " \
                "variable is set"
            self.mwer_command = mwerSegmenter_root + "/mwerSegmenter"

    def __call__(self, prediction: str, reference_sentences: List[str]) -> List[str]:
        """
        Segments the prediction based on the reference sentences using the edit distance algorithm.
        """
        tmp_pred = tempfile.NamedTemporaryFile(mode="w", delete=False)
        tmp_ref = tempfile.NamedTemporaryFile(mode="w", delete=False)
        try:
            tmp_pred.write(prediction)
            tmp_ref.writelines(ref + '\n' for ref in reference_sentences)
            tmp_pred.flush()
            tmp_ref.flush()
            subprocess.run([
                self.mwer_command,
                "-mref",
                tmp_ref.name,
                "-hypfile",
                tmp_pred.name,
                "-usecase",
                "1"])
            # mwerSegmenter writes into the __segments file of the current working directory
            with open("__segments") as f:
                return [line.strip() for line in f.readlines()]
        finally:
            tmp_pred.close()
            tmp_ref.close()
            os.unlink(tmp_pred.name)
            os.unlink(tmp_ref.name)
            os.unlink("__segments")


class SegmentLevelDelayElapsed:
    """
    The computational-aware delay (or elapsed time) for a given word is the sum of the ideal delay
    and the computation cost (or runtime) required for generating it.
    In SimulEval, the computation cost is computed as the difference between the system timestamp
    when a word is generated and the system timestamp at which the processing of the audio segment
    started.
    As the start time is reset only between each processed audio (i.e., between audio segments in
    SimulST), the elapsed time of each word is the sum of the computational time required to
    generate all the words up to it, rather than just the time required for that word.
    Ultimately, this results in the elapsed time growing indefinitely, especially in the
    streaming settings where the audio segments are long and the returned values are
    unrealistically large.

    Therefore, given that the SimulEval elapsed times are [e_1, ..., e_n], the ideal delays are
    [d_1, ..., d_n], and the SimulEval computation cost are [t_1, ..., t_n], where e_i = d_i + t_i,
    we compute the correct computational-aware latency (elapsed times) as:
        d_i + (t_i - t_i-1) = d_i + (e_i - d_i) - (e_i-1 - d_i-1) = e_i - e_i-1 + d_i-1
    """
    def __init__(self, simuleval_instance: SimulEvalLogInstance):
        self.latency_unit = simuleval_instance.latency_unit
        self.simuleval_delays = simuleval_instance.delays
        self.simuleval_elapseds = simuleval_instance.elapsed
        self.processed_latency_units = 0
        self.prev_delay = None
        self.prev_elapsed = None
        self.prev_stream_delay = None
        self.prev_stream_elapsed = None

    def __call__(
            self,
            sentence_predicted: str,
            sentence_start_ms: float) -> Tuple[List[float], List[float]]:
        prediction_len = SimulEvalLogInstance.string_to_len(
            sentence_predicted.strip(), self.latency_unit)
        stream_delays = []
        stream_elapseds = []
        delays = self.simuleval_delays[
                 self.processed_latency_units:self.processed_latency_units + prediction_len]
        elapseds = self.simuleval_elapseds[
                   self.processed_latency_units:self.processed_latency_units + prediction_len]
        assert len(delays) == len(elapseds)
        for delay, elapsed in zip(delays, elapseds):
            # For the first element of the whole audio, as the elapsed time computed by SimulEval
            # is the delay (which includes potential audio before the first sentence) plus the
            # computational cost incurred so far in processing the current audio, the streaming
            # elapsed is just the elapsed time.
            if self.prev_elapsed is None:
                stream_elapsed = elapsed
            # If the elapsed time of this element is the same of the previous one, the
            # computational cost has to be copied, while the ideal delay has to be updated. Since
            # the elapsed time is the sum of the computational cost and the ideal delay, we isolate
            # the computational cost by subtracting the previous ideal delay from the previous
            # elapsed, and we add the current ideal delay.
            elif elapsed == self.prev_elapsed:
                stream_elapsed = self.prev_stream_elapsed - self.prev_stream_delay + delay
            else:
                stream_elapsed = elapsed - self.prev_elapsed + self.prev_delay
            # Subtract the offset (start time of the segment)
            stream_delay = delay - sentence_start_ms
            stream_elapsed = stream_elapsed - sentence_start_ms
            stream_delays.append(stream_delay)
            stream_elapseds.append(stream_elapsed)
            self.prev_elapsed = elapsed
            self.prev_delay = delay
            self.prev_stream_delay = stream_delay
            self.prev_stream_elapsed = stream_elapsed
        self.processed_latency_units += prediction_len
        return stream_delays, stream_elapseds


def parse_simuleval_instances(filename: str, latency_unit: str) -> Dict[str, SimulEvalLogInstance]:
    """
    Reads a SimulEval instances log file and returns a dictionary in which the keys
    are the names of for each-instance wav file.
    """
    instances = {}
    with open(filename, "r") as f:
        for line in f:
            instance = json.loads(line.strip())
            wav_name = os.path.basename(instance["source"][0])
            instances[wav_name] = SimulEvalLogInstance(instance, latency_unit=latency_unit)
    return instances


def parse_references(
        reference_filename: str,
        audio_yaml_filename: str) -> Dict[str, List[ReferenceSentenceDefinition]]:
    """
    Reads two files:
     - a reference file containing the segment-level textual targets, one line for each segment,
     - a yaml file containing the audio information (duration and offset/start time) for each.
       segment
    The information is returned in a dictionary where the keys are the name of the wav files
    and the values are the list of reference sentences for each of those files.
    """
    with open(audio_yaml_filename) as f:
        sentence_definitions = yaml.load(f, Loader=yaml.FullLoader)
    with open(reference_filename) as f:
        sentences = f.readlines()
    assert len(sentence_definitions) == len(sentences), \
        f"Number of reference sentences ({len(sentences)}) and sentence definitions " \
        f"({len(sentence_definitions)}) should be the same."
    references = {}
    for sentence, definition in zip(sentences, sentence_definitions):
        wav_name = os.path.basename(definition["wav"])
        if wav_name not in references:
            references[wav_name] = []
        references[wav_name].append(ReferenceSentenceDefinition(
            sentence.strip(), definition["offset"], definition["duration"]))
    return references


def resegment_instances(
        predictions: Dict[str, SimulEvalLogInstance],
        references: Dict[str, List[ReferenceSentenceDefinition]]) -> List[SimulEvalLogInstance]:
    """
    Resegments the streaming instances into segment-level instances to be used for the metrics
    calculation. These instances follow the structure of the instances returned by SimulEval.
    """
    instances = []
    mwer_segmenter = MwerSegmenter()
    for wav, ref_sentences in references.items():
        predicted_log_instance = predictions[wav]
        resegmented_predictions = mwer_segmenter(
            predicted_log_instance.prediction, [sent.content for sent in ref_sentences])
        assert len(ref_sentences) == len(resegmented_predictions), \
            f"The number of resegmented predictions ({len(resegmented_predictions)}) does " \
            f"not match the number of reference sentences ({len(ref_sentences)}) for {wav}"
        delays_processor = SegmentLevelDelayElapsed(predicted_log_instance)
        for sentence_ref, sentence_pred in zip(ref_sentences, resegmented_predictions):
            if len(sentence_pred.strip()) == 0:
                stream_delays, stream_elapseds = [], []
            else:
                stream_delays, stream_elapseds = delays_processor(
                    sentence_pred, sentence_ref.start_time * 1000)
            instances.append(SimulEvalLogInstance({
                "reference": sentence_ref.content.strip(),
                "source_length": sentence_ref.duration * 1000,
                "prediction": sentence_pred.strip(),
                "delays": stream_delays,
                "elapsed": stream_elapseds,
            }, latency_unit=predicted_log_instance.latency_unit))
    return instances


def evaluate_instances(resegmented_instances: List[SimulEvalLogInstance], tokenizer: str) -> None:
    """
    Computes BLEU, StreamLAAL, and computationally aware StreamLAAL relying on the built-in
    SimulEval quality and latency scorers.
    """
    ca_unaware_scorer = LAALScorer()
    ca_aware_scorer = LAALScorer(computation_aware=True)
    bleu_scorer = SacreBLEUScorer(tokenizer)
    resegmented_instances_dict = {i: ins for i, ins in enumerate(resegmented_instances)}
    ca_unaware_score = ca_unaware_scorer(resegmented_instances_dict)
    ca_aware_score = ca_aware_scorer(resegmented_instances_dict)
    bleu_score = bleu_scorer(resegmented_instances_dict)
    print(f"BLEU\tStreamLAAL\tStreamLAAL_CA\n{bleu_score}\t{ca_unaware_score}\t{ca_aware_score}")


def main(args):
    predictions = parse_simuleval_instances(args.simuleval_instances, args.latency_unit)
    references = parse_references(args.reference, args.audio_yaml)
    assert set(predictions.keys()) == set(references.keys()), \
        "References and predictions should refer to the same audio files. Instead, they refer " \
        f"to:\nReferences: {references.keys()}\nPredictions: {predictions.keys()}"
    resegmented_instances = resegment_instances(predictions, references)
    evaluate_instances(resegmented_instances, args.sacrebleu_tokenizer)


def cli_main():
    """
    Computes the StreamLAAL metric for streaming speech translation, as an extension of the LAAL
    metric for simultaneous ST introduced in `"Over-Generation Cannot Be Rewarded: Length-Adaptive
    Average Lagging for Simultaneous Speech Translation" by Papi et al. (2022)
    <https://aclanthology.org/2022.autosimtrans-1.2/>`_.

    The script gets as input:
    - the output obtained by SimulEval using the streaming agents (stored in instances.log),
    - the textual references at segment-level,
    - the audio information at segment-level (stored in a yaml file).

    For further details see the paper that introduced StreamLAAL:
    "Direct Streaming Speech-to-Text Translation with Attention-based Audio History Selection" by
    Papi et al. (2024).
    """
    print(f"StreamLAAL v{VERSION}.")
    parser = argparse.ArgumentParser(
        description="Computes the StreamLAAL metric for streaming speech translation, as an "
                    "extension of the LAAL metric for simultaneous ST. For further details, "
                    "please check the paper that introduced StreamLAAL.",
        prog="streamLAAL")
    parser.add_argument(
        "--simuleval-instances", "-siminst", required=True, type=str,
        help="Path to the file instances.log generated by SimulEval.")
    parser.add_argument(
        "--reference", "-ref", required=True, type=str,
        help="Path to the textual file containing segment-level references stored line by line.")
    parser.add_argument(
        "--audio-yaml", "-yaml", required=True, type=str,
        help="Path to the yaml file containing the segment-level audio information.")
    parser.add_argument(
        "--sacrebleu-tokenizer", "-tok",
        choices=["13a", "char", "zh", "ja-mecab", "ko-mecab", "intl", "none"], default="13a",
        help="sacreBLEU tokenizer to be used. Default: 13a (same as sacreBLEU).")
    parser.add_argument(
        "--latency-unit", choices=["word", "char"], default="word",
        help="Whether to computed latency based on words or characters. Default: word.")
    parser.add_argument(
        "--citation", "-cite", default=False, action='store_true',
        help='Print the bibtex for citation and exit.')
    args = parser.parse_args()
    if args.citation:
        print("BibTeX:\n" + _CITATION)
        exit()
    main(args)


if __name__ == "__main__":
    cli_main()