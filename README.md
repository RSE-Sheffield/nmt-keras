# deepQuest
### neural-based Quality Estimation
Developed at the [University of Sheffield][1], deepQuest provides state-of-the-art models for multi-level quality estimation of neural machine translation.

## Documentation
Documentation for deepQuest2 is provided at [https://rse.shef.ac.uk/nmt-keras/](https://rse.shef.ac.uk/nmt-keras/)

## Quick Start
deepQuest2 is developed in python and can be used either as a command-line tool or as a python module.
### Installation
For quick installation instructions see here. For more detailed instructions [see the docs](https://rse.shef.ac.uk/nmt-keras/). We recommend using the python package manager, pip to install deepQuest and installing in a conda environment.
#### from GitHub repo url
Within a conda environment (for example named `conda-env`):
```shell
(conda-env)$ pip install https://github.com/RSE-Sheffield/nmt-keras
```
#### from remote whl, tar or zip
```shell
(conda-env)$ pip install https://github.com/RSE-Sheffield/nmt-keras/master/archive/something.whl
```
#### from local whl, tar or zip
Download the wheel, tar or zip from the releases tab of the [deepQuest GitHub repo](https://github.com/RSE-Sheffield/deepQuest)  to a local location and
```shell
(conda-env)$ pip install deepQuest.whl
```
#### from remote whl, tar or zip
If you need to have access to the source code for development, then we suggest git cloning the deepQuest gitHub repository and installing the package using `pip` to install from `setup.py`.
```shell
(conda-env)$ git clone https://github.com/RSE-Sheffield/deepQuest
(conda-env)$ cd deepquest
(conda-env)$ pip install -e .
```


## Attribution
If you use this, please cite:

<b>[DeepQuest: a framework for neural-based Quality Estimation][7]</b>. [Julia Ive][2], [Frédéric Blain][3], [Lucia Specia][4] (2018).

    @article{ive2018deepquest,
      title={DeepQuest: a framework for neural-based Quality Estimation},
      author={Julia Ive and Frédéric Blain and Lucia Specia},
      journal={In the Proceedings of COLING 2018, the 27th International Conference on Computational Linguistics, Sante Fe, New Mexico, USA},
      year={2018}
    }


## Acknowledgements
The development of deepQuest received funding from the [European Association for Machine Translation][5] and the [Amazon Academic Research Awards][6] program. deepQuest2 was developed with assistance from the [Research Software Engineering][8] team at the University of Sheffield.


[1]: https://www.sheffield.ac.uk
[2]: https://github.com/julia-ive
[3]: https://fredblain.org/
[4]: https://www.imperial.ac.uk/people/l.specia
[5]: http://eamt.org/
[6]: https://ara.amazon-ml.com/
[7]: http://aclweb.org/anthology/C18-1266
[8]: http://rse.shef.ac.uk
