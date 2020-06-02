# Køpsala (Copenhagen-Uppsala) IWPT2020 Enhanced Dependency Parser

Based on the [HIT-SCIR CoNLL2019 Unified Transition-Parser](https://github.com/DreamerDeo/HIT-SCIR-CoNLL2019)
(paper: [HIT-SCIR at MRP 2019: A Unified Pipeline for Meaning Representation Parsing via Efficient Training and Effective Encoding](https://www.aclweb.org/anthology/K19-2007.pdf)).

IWPT2020 Shared Task official website: <https://universaldependencies.org/iwpt20/>

## Pre-requisites

- Python 3.6
- AllenNLP 0.9.0

## Package structure

* `bash/` command pipelines and examples
* `config/` Jsonnet config files
* `metrics/` metrics used in training and evaluation
* `modules/` implementations of modules
* `toolkit/` external libraries and dataset tools
* `utils/` code for input/output and pre/post-processing

## Authors

- Artur Kulmizev
- Miryam de Lhoneux
- Elham Pejhan
- Daniel Hershcovich

## Citation

More details are available in [the following publication](https://arxiv.org/abs/2005.12094):

    @inproceedings{hershcovich2020koepsala,
        author={Hershcovich, Daniel and de Lhoneux, Miryam and Kulmizev, Artur and Pejhan, Elham and Nivre, Joakim},
        title={K{\o}psala: Transition-Based Graph Parsing via Efficient Training and Effective Encoding},
        booktitle={Proc. of IWPT Shared Task},
        month={July},
        year={2020}
    }

For further information, please contact Daniel Hershcovich: <dh@di.ku.dk>
