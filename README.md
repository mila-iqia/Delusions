
  

  

# Delusions

  

  

A `PyTorch` Implementation for experiments in **"Rejecting Hallucinated State Targets during Planning"**

  

-- an ICML 2025 conference paper by *Mingde "Harry" Zhao, Tristan Sylvain, Romain Laroche, Doina Precup, Yoshua Bengio*

[**arXiv**](https://arxiv.org/abs/2410.07096)

**BibTex**:
```latex
@inproceedings{
	zhao2025reject,
	title={Rejecting Hallucinated State Targets during Planning},
	author={Mingde Zhao, Tristan Sylvain, Romain Laroche, Doina Precup, Yoshua Bengio},
	booktitle={Forty-Second International Conference on Machine Learning (ICML)},
	year={2025},
	url={https://openreview.net/forum?id=40gBawg6LX},
	note={\url{https://arxiv.org/abs/2410.07096}},
}
```

This repo was implemented by Harry Zhao ([@PwnerHarry](https://github.com/PwnerHarry)), mostly adapted from [*Skipper*](https://github.com/mila-iqia/Skipper/) and [*DreamerV2-pytorch*](https://github.com/esteveste/dreamerV2-pytorch)

  

  

This work was initiated during Harry's Mitacs Internship at RBC Borealis (formerly Borealis AI), under the mentorship of Tristan Sylvain ([@TiSU32](https://github.com/TiSU32)).

  
## Use `common/evaluator.py` for SIMPLE integration with your agent!!!

  

## Python virtual environment configuration:


1. Create a virtual environment with `conda` or `venv`. We used Python 3.10 for `minigrid` experiments. Note that for *Dreamer* experiments, Python <=3.10 is needed for compatibility with `pip install ale-py==0.7.5` (from `pip install -r experiments/Dreamer/requirements.txt`)
  
2. Install PyTorch according to the [official guidelines](https://pytorch.org/get-started/locally/), make sure it recognizes your GPUs!

3.  `pip install -r requirements.txt` to install dependencies (for *Skipper*, *LEAP* and *Dyna* on `RandDistShift` and `SwordShieldMonster`, one shared virtual environment would be sufficient. *Dreamer* experiments would need a separate one for the distinctive requirements.)


## Check the results with `tensorboard`!

  
  

  

## For experiments, write bash scripts to call these `python` scripts:

  
  

`experiments/Skipper/run_minigrid_mp.py`: a multi-processed experiment initializer for *Skipper* variants for minigrid experiments
  
`experiments/{Skipper,Dyna}/run_minigrid.py`: a single-processed experiment initializer for *Skipper* or *Dyna* minigrid experiments

`experiments/LEAP/run_leap_pretrain_vae.py`: a single-processed experiment initializer for pretraining generator for the *LEAP* agent

`experiments/LEAP/run_leap_pretrain_rl.py`: a single-processed experiment initializer for training distance estimator (policy) for the *LEAP* agent. Provide the existing seed acquired from `run_leap_pretrain_vae.py`.

`experiments/Dreamer/dreamerv2/train.py`: for running the PyTorch DreamerV2. Use `--evaluator True --evaluator_reject True` to enable the training of and rejection by the evaluator, respectively. 

For `minigrid` experiments, please read carefully the argument definitions in `runtime.py` and pass the desired arguments.


## To control the HER variants (`minigrid` experiments):

  

  

  

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

  

  

- The Dynamic Programming (DP) solutions for `minigrid` ground truth are only compatible with deterministic experiments
