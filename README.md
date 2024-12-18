# LLM-Based Neural Network Generator

<!--
## Installation with pip
pip install git+https://github.com/ABrain-One/nn-gen
-->

## Environment
### Pip package manager
Create a virtual environment, activate it, and run the following command to install all the project dependencies: <br/> 
<strong>pip install -r requirements.txt</strong>

### Docker
All versions of this project are compatible with <a href='https://hub.docker.com/r/abrainone/ai-linux' target='_blank'>AI Linux</a> and can be run inside a Docker image: <br/> 
<strong> docker run -v /a/mm:&#x003C;nn-gen path&#x003E;/ab/gen abrainone/ai-linux bash -c "PYTHONPATH=/a/mm python train.py" </strong>

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
