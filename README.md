# <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> GPT-Driven Neural Network Generator
<sub><a href='https://pypi.python.org/pypi/nn-gpt'><img src='https://img.shields.io/pypi/v/nn-gpt.svg'/></a><br/>
short alias  <a href='https://pypi.python.org/pypi/lmurg'>lmurg</a></sub>

<img src='https://abrain.one/img/lemur-nn-gen-whit.jpg' width='25%'/>

<h3>Overview 📖</h3>

This Python-based project leverages large language models to automate the creation of neural network architectures, streamlining the design process for machine learning practitioners.

## Update of NN Dataset
Remove old version of the LEMUR Dataset and its database:
```bash
source .venv/bin/activate
pip uninstall nn-dataset -y
rm -rf db
```
Installing the stable version:
```bash
source .venv/bin/activate
pip install nn-dataset --upgrade --extra-index-url https://download.pytorch.org/whl/cu124
```
Installing from GitHub to get the most recent code and statistics updates:
```bash
source .venv/bin/activate
pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu124
```
Adding functionality to export data to Excel files and generate plots for <a href='https://github.com/ABrain-One/nn-stat'>analyzing neural network performance</a>:
```bash
source .venv/bin/activate
pip install nn-stat --upgrade --extra-index-url https://download.pytorch.org/whl/cu124
```
and export/generate:
```bash
source .venv/bin/activate
python -m ab.stat.export
```

## Environment for NN GPT Developers
### Pip package manager
Create a virtual environment, activate it, and run the following command to install all the project dependencies:
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
```

### Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be run inside a Docker image:
```bash
docker run -v /a/mm:. abrainone/ai-linux bash -c "PYTHONPATH=/a/mm python -m ab.gen.train_n_eval"
```

## Citation

The original version of this project was created at the Computer Vision Laboratory of the University of Würzburg by the authors mentioned below. If you find this project to be useful for your research, please consider citing:
```bibtex
@misc{ABrain-One.NN-GPT,
  author       = {... and Goodarzi, Arash Torabi and ... and Ignatov, Dmitry and Timofte, Radu},
  title        = {GPT-Driven Neural Network Generator},
  howpublished = {\url{https://github.com/ABrain-One/nn-gpt}},
  year         = {2024},
}
```

## Licenses

This project is distributed under the following licensing terms:
<ul><li>for neural network models adopted from other projects
  <ul>
    <li> Python code under the legacy ... license</li>
    <li> models with pretrained weights under the legacy ... license</li>
  </ul></li>
<li> all neural network models and their weights not covered by the above licenses, as well as all other files and assets in this project, are subject to the <a href="LICENSE">MIT license</a></li> 
</ul>

#### The idea of Dr. Dmitry Ignatov
