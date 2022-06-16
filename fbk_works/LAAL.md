# Length Adaptive Average Lagging

To compute the Length Adaptive Average Lagging (LAAL) metric in addition to the standard metrics provided by 
the [SimulEval tool](https://github.com/facebookresearch/SimulEval), run the following code by giving
as input the `instances.log` output file of SimulEval:
```bash
python ${FBKFAIRSEQ_HOME}/examples/speech_to_text/scripts/LAAL_metric.py instances.log
```
The result will be similar to:
```bash
LAAL: 1000
LAAL (CA): 1300 
```
where CA stands for the Computational Aware version of the LAAL metric, as for the SimulEval notation.

# Citation

```bibtex
@inproceedings{papi-et-al-2022-LAAL,
title = "Over-Generation Cannot Be Rewarded: Length-Adaptive Average Lagging for Simultaneous Speech Translation",
author = {Papi, Sara and Gaido, Marco and Negri, Matteo and Turchi, Marco},
booktitle = "Proceedings of the Third Workshop on Automatic Simultaneous Translation",
year = "2022",
month = jul,
address = "Seattle, Washington and Online",
publisher = "Association for Computational Linguistics"
}
```