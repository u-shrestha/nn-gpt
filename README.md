# <img src='https://abrain.one/img/lemur-nn-icon-64x64.png' width='32px'/> GPT-Driven Neural Network Generator

<sub><a href='https://pypi.python.org/pypi/nn-gpt'><img src='https://img.shields.io/pypi/v/nn-gpt.svg'/></a> <a href="https://pepy.tech/project/nn-gpt"><img alt="GitHub release" src="https://static.pepy.tech/badge/nn-gpt"></a><br/>
short alias  <a href='https://pypi.python.org/pypi/lmurg'>lmurg</a> 
</sub> 
<br/><br/>
<img src='https://abrain.one/img/lemur-nn-gen-whit.jpg' width='25%'/>
<h3>Overview 📖</h3>

This Python-based <a href='https://github.com/ABrain-One/nn-gpt'>NNGPT</a> project leverages large language models (LLMs) to automate the creation of neural network architectures, streamlining the design process for machine learning practitioners. It leverages various neural networks from the <a href="https://github.com/ABrain-One/nn-dataset">LEMUR Dataset</a> to fine-tune LLMs and provide insights into potential architectures during the creation of new neural network models.

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

It is also assumed that CUDA 12.6 is installed. If you have a different version, please replace 'cu126' with the appropriate version number.

## Environment for NNGPT Developers
### Pip package manager

#### Prerequisites for mpi4py package:
* On Debian/Ubuntu systems, run:
  
         sudo apt install libmpich-dev    # for MPICH
  
         sudo apt install libopenmpi-dev  # for Open MPI

* On Fedora/RHEL systems, run:
  
         sudo dnf install mpich-devel     # for MPICH
  
         sudo dnf install openmpi-devel   # for Open MPI
  
Create a virtual environment, activate it, and run the following command to install all the project dependencies:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-no-isolation.txt --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cu126
```

If there are installation problems, install the dependencies from the 'requirements.txt' file one by one.

## Update of NN Dataset
Remove an old version and install LEMUR Dataset from GitHub to get the most recent code and statistics updates:
```bash
rm -rf db
pip uninstall nn-dataset -y
pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --force --extra-index-url https://download.pytorch.org/whl/cu126
```
Installing the stable version:
```bash
pip install nn-dataset --upgrade --extra-index-url https://download.pytorch.org/whl/cu126
```
Adding functionality to export data to Excel files and generate plots for <a href='https://github.com/ABrain-One/nn-stat'>analyzing neural network performance</a>:
```bash
pip install nn-stat --upgrade --extra-index-url https://download.pytorch.org/whl/cu126
```
and export/generate:
```bash
python -m ab.stat.export
```

### Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be seamlessly executed within the AI Linux Docker container.

Installing the latest version of the project from GitHub
```bash
docker run --rm -u $(id -u):ab -v $(pwd):/a/mm abrainone/ai-linux bash -c "[ -d nn-gpt ] && git -C nn-gpt pull || git -c advice.detachedHead=false clone --depth 1 https://github.com/ABrain-One/nn-gpt"
```

Running script
```bash
docker run --rm -u $(id -u):ab --shm-size=16G -v $(pwd)/nn-gpt:/a/mm abrainone/ai-linux bash -c "python -m ab.gpt.TuneNNGen_8B"
```

The recently added dependencies might be missing in the <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a>. In this case, you can create a container from the Docker image ```abrainone/ai-linux```, install the missing packages (preferably using ```pip install <package name>```), and then create a new image from the container using ```docker commit <container name> <new image name>```. You can use this new image locally or push it to the registry for deployment on the computer cluster.

## Use

- **`ab.gpt.NNAlter*.py`** – Generates modified neural network models.  
  Use the `-e` argument to set the number of epochs for the initial CV model generation.

- **`ab.gpt.NNEval.py`** – Evaluates the models generated in the previous step.

- **`ab.gpt.TuneNNGen*.py`** – Performs fine-tuning and evaluation of an LLM. For evaluation purposes, the LLM generates neural network models, which are then trained to assess improvements in the LLM’s performance on this task. The -s flag allows skipping model generation for the specified number of epochs.

<a href='https://huggingface.co/ABrain'><strong>Pretrained LLM weights</strong></a>

## Citation

The original version of this project was created at the Computer Vision Laboratory of the University of Würzburg by the authors mentioned below. If you find this project to be useful for your research, please consider citing our articles for NNGPT framework and <a target='_blank' href='https://arxiv.org/pdf/2504.06006'>hyperparameter tuning</a>:
```bibtex
@article{ABrain.NNGPT,
  title        = {NNGPT: Rethinking AutoML with Large Language Models},
  author       = {Kochnev, Roman and Khalid, Waleed and Uzun, Tolgay Atinc and Zhang, Xi and Dhameliya, Yashkumar Sanjaybhai and Qin, Furui and Ignatov, Dmitry and Timofte, Radu},
  year         = {2025}
}

@article{ABrain.HPGPT,
  title={Optuna vs Code Llama: Are LLMs a New Paradigm for Hyperparameter Tuning?},
  author={Kochnev, Roman and Goodarzi, Arash Torabi and Bentyn, Zofia Antonina and Ignatov, Dmitry and Timofte, Radu},
  journal={arXiv preprint arXiv:2504.06006},
  year={2025}
}
```
## Licenses

This project is distributed under the following licensing terms:
<ul><li>models with pretrained weights under the legacy <a href="https://github.com/ABrain-One/nn-dataset/blob/main/Doc/Licenses/LICENSE-DEEPSEEK-LLM-V2">DeepSeek LLM V2</a> license</li>
<li> all neural network models and their weights not covered by the above licenses, as well as all other files and assets in this project, are subject to the <a href="LICENSE">MIT license</a></li> 
</ul>

#### The idea and leadership of Dr. Ignatov
