# PHY241 Particle Discovery Project

Experiment design for discovering a 5 GeV meson decaying to two muons.

This repository contains the analysis notebook used for the PHY241 group project.

Contributors: Sandrin Hunkeler, Manasi Tiwari, Elisabeth Giryes

## Project idea

The workflow is split into two parts:

1. Use Monte Carlo samples with 10k signal and 10k background events to design the event selection.
2. Use the resulting efficiencies to estimate how long the experiment must run for a 5 sigma discovery.

The Monte Carlo samples are only for developing and validating the selection. They are not the same as the realistic yearly yield, which is much smaller.

## Main results

From the current notebook:

- 7-feature BDT test accuracy: 96.44%
- Signal efficiency: 0.9774 +/- 0.0021
- Background efficiency: 0.0487 +/- 0.0031
- Mean significance for 1 year: 11.31 sigma
- Fraction of 1-year toys above 5 sigma: 100.0%
- Minimum duration for 95% discovery probability: 5.0 months

## Repository contents

- `particle_discovery.ipynb` - main analysis notebook
- `data/` - signal and background Monte Carlo samples
- `plots/` - generated figures

## How to run the analysis

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook particle_discovery.ipynb
```

## Notes on the analysis

- Fisher scores are used to rank the input features.
- Rectangular cuts provide a simple baseline selection.
- The BDT is trained with a train/test split so the reported efficiencies are not biased by training on the same events.
- The mass fit uses a Gaussian signal model plus an exponential background model.
- Discovery significance is estimated with toy experiments and Wilks' theorem.

