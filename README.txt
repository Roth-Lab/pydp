# Overview

PyDP is library for implementing Dirichlet Process mixture models (DPMM). The goal of PyDP is to provide a pure Python 
implementation of various algorithms for working DPMMs. As a design choice PyDP should have no dependencies on any 
libraries which are not supported by the [PyPy](http://pypy.org) Python interpreter. 

# License

PyDP is licensed under the GPL v3, see the LICENSE.txt file for details.

# Versions

## 0.2.3

* Fixed bug in mpear

## 0.2.2

* Added code for vector distributions

* Added code for clustering using MPEAR

## 0.2.1

* Fixed a bug in concentration sampler

* Fixed log_beta function to check if parameters are <= 0 and return -inf if so


## 0.2.0

* Changed the interface for AtomSampler to take cells instead of partitions.

* Added global parameter updating.

* Updated density interface to use caching.

* Added some new proposal functions.

## 0.1.5

* Fixed error in concentration sampler due to using the wrong parameterisation of the gamma prior. 

## 0.1.4

* Fixed underflow issue in precision update for Gaussian model.

## 0.1.3

* Added code for Gaussian models.

* Added wrapper class for DP sampler. 

## 0.1.2

* Added GPL license informtation.

# Installation

Installation is the standard `python setup.py install`.

## Dependencies

### Required

* None

### Optional

* [SymPy](http://sympy.org/en/index.html) >= 0.7.1 - Used for some of the diagnostic tools to compute the chi-square 
													 distribution.
