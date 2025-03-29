# <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> GPT-Driven Neural Network Generator
<sub><a href='https://pypi.python.org/pypi/nn-gpt'><img src='https://img.shields.io/pypi/v/nn-gpt.svg'/></a><br/>
short alias  <a href='https://pypi.python.org/pypi/lmurg'>lmurg</a></sub>

<img src='https://abrain.one/img/lemur-nn-gen-whit.jpg' width='25%'/>

<h3>Overview 📖</h3>

This Python-based project leverages large language models to automate the creation of neural network architectures, streamlining the design process for machine learning practitioners.

## Create and Activate a Virtual Environment (recommended)
For Linux/Mac:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
For Windows:
   ```bash
   python3 -m venv .venv
   .venv\Scripts\activate
   ```

All subsequent commands are provided for Linux/Mac OS. For Windows, please replace ```source .venv/bin/activate``` with ```.venv\Scripts\activate```.
It is also assumed that CUDA 12.6 is installed. If you have a different version, please replace 'cu126' with the appropriate version number.

## Environment for NNGPT Developers
### Pip package manager
Create a virtual environment, activate it, and run the following command to install all the project dependencies:
```bash
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

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
pip install nn-dataset --upgrade --extra-index-url https://download.pytorch.org/whl/cu126
```
Installing from GitHub to get the most recent code and statistics updates:
```bash
source .venv/bin/activate
pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu126
```
Adding functionality to export data to Excel files and generate plots for <a href='https://github.com/ABrain-One/nn-stat'>analyzing neural network performance</a>:
```bash
source .venv/bin/activate
pip install nn-stat --upgrade --extra-index-url https://download.pytorch.org/whl/cu126
```
and export/generate:
```bash
source .venv/bin/activate
python -m ab.stat.export
```

### Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be run inside a Docker image:
```bash
docker run -v /a/mm:. abrainone/ai-linux bash -c "PYTHONPATH=/a/mm python -m ab.gpt.train_n_eval"
```

The recently added dependencies might be missing in the <b>AI Linux</b>. In this case, you can create a container from the Docker image ```abrainone/ai-linux```, install the missing packages (preferably using ```pip install <package name>```), and then create a new image from the container using ```docker commit <container name> <new image name>```. You can use this new image locally or push it to the registry for deployment on the computer cluster.

## Usage

Use `initial_generation_chg.py` to generate initial modified CV models, specify by argument `-e` to determine the number of epochs for initial CV model generation.

Use `finetune_nn_gen.py` to perform generation and evaluation of CV model, evaluate and fine-tune the LLM. Use argument `-s` to colaborate with `generate.py`, with `-s` for number of epochs to skip the CV model generation.

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
