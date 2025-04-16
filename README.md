# Precautionary saving and portfolio choice with joint income and equity return processes

This repository contains the Python and $\LaTeX$ code used to solve the model and typeset the paper.

## Running the code

The model is solved using the endogenous gridpoints method, and uses the equiprobable discretization method for the multivariate lognormal distribution available in the `HARK` toolkit. Pending some changes to the tool, the version of `HARK` installed and used in the code is available on my fork of the `HARK` repository by `econ-ark`. To efficiently run the code, I recommend creating a fresh Python environment, and install the necessary libraries (`jupyter`, `numpy`, `scipy`, `matplotlib`, and `econ-ark`). Particularly, run the following command to install the appropriate version of the `HARK` toolkit:

```
pip install git+https://github.com/econ-ark/HARK.git
```

**NOTE**: These instructions are subject to change, as and when the `HARK` toolkit is updated.

The $\LaTeX$ code is self-sufficient, and should work with any up-to-date $\TeX$ distribution.

## Files

The code used to solve the model and generate the plots in the paper are available in the folder titled `Code`. While the code can be run for replication purposes using `portfolio_choice.py`, the file `consumer.py` contains the definition of the model class, which is called upon to create an instance of an agent's problem and solve it. The draft of the paper is available in the folder titled `TeX`, by the name `precautionary-saving-portfolio-choice.pdf`.
