# ABLATOR Template Project

This repository is inteded to help [ABLATOR](https://ablator.org) users to package their experiment.

It uses [poetry](https://python-poetry.org/), but the tool is not required to use or edit this Template to fit your needs. However we recommend to install poetry to help you manage your depedencies. e.g. `poetry add library`

## Why Package?
There are several reasons to package your project.

1. You can import modules from within your project with absolute imports that is more robust to enviroment changes.
2. You can manage the depedencies of your project such that it cna be used for different enviroments, e.g. by others or even yourself such as changing your run-time enviroment from `colab` to a GPU workstation.

3. It allows you to keep track of changes and perform better revision control. e.g. `v0.1.0` of your project can be the main implementation of your method while `v0.1.0c` is the experimental version of your `contrastive-learning` experiments.

All of the above help with:
1. Debugging, such as with `vscode` where we provide some default vscode configuration in this project to help you get started. You can use the vscode debugger and set breakpoints in your code to debug for example, whether gradient updates are happening as you expect.
2. Distributing your project to run in a distributed fashion. As a GPU cluster will be composed of several enviroments, your project will need to be installable and executable in all of them and `Dockerized`.

## Prerequisites

Create a virtual enviroment, such as with conda. Make sure you use the correct python version for the [ABLATOR](https://ablator.org) version you are using.

e.g.  `conda create --name mycoolproject python=3.10`


### Modify

Use the template ![Template Use](assets/image.png)


You should modify [`pyproject.toml`](pyproject.toml)

To replace:

1. `name = "ablator-skeleton"`

2. `authors = ["Your Name <you@example.com>"]`

3. `packages = [{include = "ablator_skeleton"}]` (project directory name)

and rename [`ablator_skeleton`](ablator_skeleton) to the choice of your name (consistent to `pyproject.toml`). We will be using `ablator_skeleton` for this Guide, but you will need to adjust the commands specific to your modifications.


## Usage

After modifying `pyproject.toml` use this project you will need to first install it.

`pip install -e .`

Using `-e` option allows for the changes to reflect in a `live` fashion from your installation directory, without having to re-run `pip install`.

### Running

You can run your experiment using `python -m ablator_skeleton`. The entry-point of your script will be [`__main__.py`](ablator_skeleton/__main__.py). We provide some default functionality and an example that you will need to modify to your use case.

For Prototyping:

`python -m ablator_skeleton`

For Distributed Training:

`python -m ablator_skeleton --mp mock_param`


### (Optional) Dockerize

Modify [`Dockerfile`](Dockerfile) and run:

```bash
cd ablator-skeleton
docker build . -t skeleton
docker run -it skeleton
```


**NOTE** working Docker installation required.

You must run `docker build` every time you modify your project.

**WARNING** The containers are ephemeral, any data stored inside the container will be deleted upon termination.

The container can be distributed and executed on any hardware.

### Designing your Experiment

Several examples on the use-cases of ablator are provided in [ABLATOR](https://ablator.org). In this repository we provide a `MOCK` model example intended to be a **no** use-case model that is easy to understand.




