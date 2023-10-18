# INES Test Suite Evaluation (WMT 2023)
Code to evaluate MT systems on the INclusive Evaluation Suite (INES).

## INES Evaluation

We release the code the FBK participation to the WMT Test Suite shared subtask: [**INES_eval.py**](../examples/speech_to_text/scripts/gender/INES_eval.py).
It allows to assess the ability of MT systems to generate inclusive language forms over non-inclusive ones when translating from German into English on the [**INES test set**](https://mt.fbk.eu/ines/).


For systems run on the INES test suite, the evaluation script "INES_eval.py" computes:

* **inclusivity_index** scores (INES official metric)

* **terms coverage** and **gender accuracy** scores (additional metrics)


### Usage

To work correctly, the script requires Python 3.

The script requires two mandatory arguments:

	--input FILE 
    --tsv-definition FILE

Namely, the output of the system you want to evaluate and the [**INES.tsv**](https://mt.fbk.eu/ines/) file (the Gold Standard). Note that the output must be tokenized (e.g. with Moses' tokenizer.perl)

You can run "INES_eval.py --help" to get a list of the parameters taken by the script.
The script computes terms coverage and gender accuracy if requested as facultative argument.

Example Usage

    python3 INES_eval.py --input MT OUTPUT FILE --tsv-definition INES.tsv


## 📍Citation

If you use this code and for more information, please refer to:

```bibtex
@inproceedings{savoldi-etal-2023-test,
    title = {{Test Suite Task: Evaluation of Gender Fairness in MT with MuST-SHE and INES}},
    author = {Savoldi, Beatrice  and Gaido, Marco  and Negri, Matteo and Bentivogli, Luisa},
    booktitle = {Proceedings of the 8th International Conference on Machine Translation (WMT 2023)},
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
}
```
