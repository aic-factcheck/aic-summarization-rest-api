# aic-summarization-rest-api
This repository is a collection of minimal code to generate summarizations using models developed at [AIC](https://www.aic.fel.cvut.cz) as well as simple REST API. We currently provide only code for network predictions (not training).
## Usage
Install required modules:
```bash
pip install -r requirements.txt
```
Then run the server, providing a configuration file:
```bash
python api.py cfg/mbart_headline.json
```

To run on CUDA use `--device` command line parameter:
```bash
python api.py --device cuda cfg/mbart_headline.json
```

Get help:
```bash
python api.py --help
```

## Models
We provide multilingual models finetuned on a news corpus focussing on the Czech language. Czech datasets involve [SumeCzech](https://ufal.mff.cuni.cz/sumeczech) and a proprietary dataset kindly provided by [Czech News Agency](https://www.cncenter.cz). The models are of two kinds:
* models trained to generate a *headline* based on a concatenation of the article abstract and the main text,
* models trained to generate an *abstract* based on a concatenation of the headline and the main text.

More details on models, data, and training are given in HuggingFace database:
* [MBART25 headline model](https://huggingface.co/krotima1/mbart-at2h-cs)
* [M2M abstract model](https://huggingface.co/ctu-aic/m2m100-418M-multilingual-summarization-multilarge-cs)
* [MBART25 abstract model](https://huggingface.co/ctu-aic/mbart25-multilingual-summarization-multilarge-cs)
* [MT5 abstract model](https://huggingface.co/ctu-aic/mt5-base-multilingual-summarization-multilarge-cs)

