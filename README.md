# Constraining Sequential Model Editing with Editing Anchor Compression

This repository contains the source code for the paper:
- [Constraining Sequential Model Editing with Editing Anchor Compression](https://arxiv.org/pdf/2503.00035.pdf). <br>
  Hao-Xiang Xu, Jun-Yu Ma, Zhen-Hua Ling, Ningyu Zhang Jia-Chen Gu<br>
  _NAACL 2025 Findings_ <br>

## Overview
In response to the challenge of hallucinations in the output of LLM due to false or outdated knowledge, **model editing** has received a lot of attention due to its low resource consumption. However, current model editing methods significantly compromise the general abilities of LLMs as the number of edits increases, and this trade-off poses a substantial challenge to the continual learning of LLMs.

In this paper, we first analyze that the factor affecting the general abilities in sequential model editing, and then propose a framework termed Editing Anchor Compression (EAC).

<img src="https://github.com/famoustourist/EAC/blob/main/definition.jpg" width=80%>

## Datasets
The datasets are included in `data/`. There are three folders:
* `edited-data`: The data used to edit the model. In this article, we use the ZsRE dataset to edit the model.
* `task-data`: The data used in downstream tasks.

The whole data directory is as follows:
```bash
data/
    |__ edited-data 
        |__ zsre.json
    |__ task-data
        |__ test-NLI.tsv
        |__ test-OpenDomainQA.jsonl
        |__ test-SentimentAnalysis.tsv
        |__ test-summarization.json
```
You can download these datasets here. [[Google Drive]](https://drive.google.com/drive/folders/1isrBQ_8MTvbP1T8BqregyDy-t7bLFmPF?usp=sharing).

## Prepare the environment

### Requirements

**Note: Please use Python 3.9+**
To get started, simply install conda and run:

```shell
git clone https://github.com/famoustourist/EAC.git
conda create -n EAC python=3.9.7
...
pip install -r requirements.txt
```

### Models
All models are putted in `hugging_cache/<model_name>` (model_name=gpt2-xl, llama3-8b, or llama2-13b).

These could be changed in `hparams/<method_name>/`.

## Evaluation
Eight different downstream task evaluation metrics are as follows

- `Natural language inference (NLI)`: accuracy of two-way classification
- `Open-domain QA`: exact match(EM) with the reference answer after minor normalization
- `Summarization`: the average of ROUGE-1, ROUGE-2 and ROUGE-L
- `Sentiment analysis`: accuracy of two-wayclassification

GPT-2 XL(1.5B), LLaMA-3(8B), LLaMA-2(13B) are used for editing.

- These model editing methods are used in our paper as follows:
  - [ROME](https://github.com/kmeng01/rome): Kevin Meng et al. Locate and Edit
  - [MEMIT](https://github.com/kmeng01/memit): Kevin Meng et al. Locate and Edit


### Running the evaluation
If you want to evaluate the performance of the pre-edit model on various downstream tasks (e.g. evaluating task NLI), run:
```bash
python test-task.py task
python test-task.py NLI
```
`task`: The name of the task you want to evaluate，you can choose from: **NLI**, **OpenDomainQA**, **SentimentAnalysis**, **summarization**.

If you want to evaluate the performance of the edited model on various downstream tasks after the introduction of EAC (e.g. evaluating task NLI with method ROME-EAC), run:
```bash
python test-task-after.py task mode method sample_begin sample_end sample_step
python test-task-after.py NLI Instance-Sequential ROME-EAC 200 204 1
```
`mode`: The mode of editing you want to use，you can choose from: **Instance-Sequential**.

`method`：The editing method you want to use，you can choose from: **ROME-EAC**, **MEMIT-EAC**.

`sample_begin`：The number at the beginning of the sample you selected in the dataset.

`sample_end`：The number at the end of the sample you selected in the dataset.

`sample_step`: One sample is selected every sample_step sample.

## Citation
If you use this code and dataset, please cite our paper:
```bibtex
@article{xu2025constraining,
  title={Constraining Sequential Model Editing with Editing Anchor Compression},
  author={Xu, Hao-Xiang and Ma, Jun-Yu and Ling, Zhen-Hua and Zhang, Ningyu and Gu, Jia-Chen},
  journal={arXiv preprint arXiv:2503.00035},
  year={2025}
}
```

### Related Projects
- [EasyEdit](https://github.com/zjunlp/EasyEdit)
- [ROME](https://github.com/kmeng01/rome)
- [MEMIT](https://github.com/kmeng01/memit)

We express sincere gratitude to [EasyEdit](https://github.com/zjunlp/EasyEdit), [ROME](https://github.com/kmeng01/rome) and [MEMIT](https://github.com/kmeng01/memit) as we have utilized portions of their source code in our project.

