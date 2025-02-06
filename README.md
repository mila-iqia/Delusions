
  

# Delusions

  

  

A `PyTorch` Implementation for experiments in

  

  

**"Rejecting Hallucinated State Targets during Planning"**

  

authored by *Mingde "Harry" Zhao, Tristan Sylvain, Romain Laroche, Doina Precup, Yoshua Bengio*

  
  

[arXiv](https://arxiv.org/abs/2410.07096)

  

  

This repo was implemented by Harry Zhao ([@PwnerHarry](https://github.com/PwnerHarry)), mostly adapted from [Skipper](https://github.com/mila-iqia/Skipper/)

  
  

This work was done during Harry's Mitacs Internship at Borealis AI (RBC), under the mentorship of Tristan Sylvain ([@TiSU32](https://github.com/PwnerHarry)).

  

  

## Python virtual environment configuration:

  

1. Create a virtual environment with conda or venv (we used Python 3.10)

  

  

2. Install PyTorch according to the [official guidelines](https://pytorch.org/get-started/locally/), make sure it recognizes your accelerators

  

  

3.  `pip install -r requirements.txt` to install dependencies

  

  

## To check the results with `tensorboard`:

  

  

`tensorboard --logdir=tb_records`

  

  

## For experiments, write bash scripts to call these `python` scripts:

  

  

`run_minigrid_mp.py`: a multi-processed experiment initializer for *Skipper* variants

  

  

`run_minigrid.py`: a single-processed experiment initializer for *Skipper* variants

  

  

`run_leap_pretrain_vae.py`: a single-processed experiment initializer for pretraining generator for the *LEAP* agent

  

  

`run_leap_pretrain_rl.py`: a single-processed experiment initializer for pretraining distance estimator (policy) for the *LEAP* agent

  

  

Please read carefully the argument definitions in `runtime.py` and pass the desired arguments.

  

  

## To control the HER variants:

  

  

Use `--hindsight_strategy` to specify the hindsight relabeling strategy. The options are:

  

  

-  `future`: same as *"future"* variant in paper

  

-  `episode`: same as *"episode"* variant in paper

  

-  `pertask`: same as *"pertask"* variant in paper

  

-  `future+episode`: correspond to *"E"* variant in paper

  

-  `future+pertask`: correspond to *"P"* variant in paper

  

-  `future+episode@0.5`: correspond to *"(E+P)"* variant in paper, where `0.5` controls the mixture ratio of `pertask`

  

  

To use the "generate" strategy for estimator training, use `--prob_relabel_generateJIT` to specify the probability of replacing the relabeled target:

  

-  `--hindsight_strategy future+episode --prob_relabel_generateJIT 1.0`: correspond to *"G"* variant in paper

  

-  `--hindsight_strategy future+episode --prob_relabel_generateJIT 0.5`: correspond to *"(E+G)"* variant in paper

  

-  `--hindsight_strategy future+episode@0.333 --prob_relabel_generateJIT 0.25`: correspond to *"(E+P+G)"* variant in paper

  
  

## To choose environment and training settings:

  

-  `--game SwordShieldMonster --size_world 12 --num_envs_train 50`: `game` can be switched with `RandDistShift (RDS)` and `size_world` should >= 8

  
  

## Extras

  

- There is a potential `CUDA_INDEX_ASSERTION` error that could cause hanging at the beginning of the *Skipper *runs. We don't know yet how to fix it

  

- The Dynamic Programming solutions for environment ground truth are only compatible with deterministic experiments
