[![hermespy Actions Status](https://github.com/barkhausen-institut/hermespy/workflows/hermespy/badge.svg)](https://github.com/barkhausen-institut/hermespy/actions)


# HermesPy

HermesPy (Heterogeneous Radio Mobile Simulator - Python) is a semi-static link-level simulator based on time-driven mechanisms.

It provides a framework for the link-level simulation of a multi-RAT wireless link, consisting of
multiple transmit and receive modems, which may operate at different carrier frequencies. Besides
simulating individual transmission links, HermesPy allows the analysis of both co-channel and
adjacent-channel interference among different communication systems.

You can find an introductory video here: https://www.barkhauseninstitut.org/opensource/hermespy

# Features

The curent release "Platform Release" serves as a platform for joint development. Beside some core functionality, that can be found beside the [release plan](FEATURES.md), its main focus is to provide a software architecture that can be easily extended.

# Installation

- `git clone <this-repo>`
- Change to `hermespy/`

**Windows users:**
- `conda create -n <envname> python=3.7` (can be omitted for ubuntu users)
- `conda activate <envname>` (can be omitted for ubuntu users)
- `conda install pip` (can be omitted for ubuntu users)
- `pip install -r requirements.txt`

**Ubuntu users**:
- Ensure `python` is linked to python3.7
- `python -m venv env`
- `. env/bin/activate`
- `pip install -r requirements.txt`
 

Since the [Quadriga channel model v2.1.30](https://quadriga-channel-model.de/) is supported by HermesPy, some preliminary steps need to be taken. It can be run with either Octave or matlab. For **octave**, under Windows, you need to set the environemnt variable that tells python where to find octave by calling

```
setx PATH "%PATH%;<path-to-octave-cli-dir>
```

and install `oct2py` via `pip install oct2py` (Ubuntu sets the environment variable automatically).

If you want to run **matlab**, you must use the `matlab.engine` package proided by Matlab. Refer to [this](https://de.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html) link.

# Running the simulator

1. Activate your virtual environment:
   1. **Windows**: `conda activate <envname>`
   2. **ubuntu**: `. env/bin/activate` 
2. The simulator can be called by `python hermes.py <param> <output_dir>`.

The main execution file is `hermes.py -p <settings-directory -o <results-directory>`. Both command line parameters are optional arguments. `<settings-directory>` contains settings files for the simulation and `<results-directory>` tells the simulator where to put the result files. Both are relative paths and have default values namely `_settings` and `results`.

`settings_general.ini` contains all the parameters that describe the simulation flow and `settings_scenario.ini` contains the parameters that describe the simulation scenario (modems,
channel). Depending on the scenario, some technology specific parameter files must also be
defined. For further information on the parameters, refer to the simulation files.

## Quadriga

Set `Multipath=QUADRIGA` for **all channels** in `settings_scenario.ini`. The Quadriga channel model will then use a combination of the values provided in all `settings_*.ini` files. Quadriga-specific values are located in the `settings_quadriga.ini`. **Check before first use.**

**Important note:** As SNR-Values, use `SNR=CUSTOM`. Setting this value to custom, values provided for `snr_vector` are interpreted as actual noise variance (in dB/Hz). Check test files for actual values.

**Important note**: If you decide to run with matlab/octave, ensure you installed matlab/octave and octpy/matlab.engine. Otherwise the program will crash. Although some catches are implemented, this can lead to errors.

## Running the tests

Tests are run by calling `pytest` in the root directory. They test for

- type linting using [mypy](mypy.readthedocs.io/)
- basic checks on pep8
- unit tests. 

**Attention:** Since running the integrity tests might take more than an hour, we advise to run them separately. Simply call `python run_integrity_tests.py`. If you don't want to check on mypy and pep8 and only want to chceck if the unit tests pass, call `python run_unit_tests.py`. No tests cover the use case of quadriga.

# Documentation

Documentation can be found [here](https://barkhausen-institut.github.io/hermespy).
Quadriga documentation can be found in **hermes/docssource**.

# Authors

* [Andre Noll Barreto](https://gitlab.com/anollba)
* [Tobias Kronauer](https://github.com/tokr-bit)

# Acknowledgments

* The structure of HermesPy was built based on the Hermes Matlab simulator developped at INDT, in
Manaus and Bras??lia.

# Copyright
Copyright (C) 2020 Barkhausen Institut gGmbH

