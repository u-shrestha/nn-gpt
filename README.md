# LLM-Based Neural Network Generator

<h3>Overview 📖</h3>

This Python-based project leverages large language models to automate the creation of neural network architectures, streamlining the design process for machine learning practitioners.

## Installation of the Latest Version of the NN-Dataset
The NN-Gen project relies on the NN-Dataset. The most recent version of the NN-Dataset can be installed directly from GitHub:
```bash
pip install git+https://github.com/ABrain-One/nn-dataset --upgrade --force
```

## Environment for NN-Gen Developers
### Pip package manager
Create a virtual environment, activate it, and run the following command to install all the project dependencies:
```bash
pip install -r requirements.txt
```

### Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be run inside a Docker image:
```bash
docker run -v /a/mm:<nn-gen path> abrainone/ai-linux bash -c "PYTHONPATH=/a/mm python -m ab.gen.train_n_eval"
```

## Citation

The original version of this project was created at the Computer Vision Laboratory of the University of Würzburg by the authors mentioned below. If you find this project to be useful for your research, please consider citing:
```bibtex
@misc{ABrain-One.NN-Gen,
  author       = {Goodarzi, Arash and ...  and Ignatov, Dmitry and Timofte, Radu},
  title        = {LLM-Based Neural Network Generator},
  howpublished = {\url{https://github.com/ABrain-One/nn-gen}},
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
<li> all neural network models and their weights not covered by the above licenses, as well as all other files and assets in this project, are subject to the <a href="LICENSE.md">MIT license</a></li> 
</ul>

#### The idea of Dr. Dmitry Ignatov
