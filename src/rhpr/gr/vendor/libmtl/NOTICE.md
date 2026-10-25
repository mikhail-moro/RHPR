Uncertainty weighting adapted from median-research-group/LibMTL, LibMTL/weighting/UW.py.
Source: https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/UW.py
Retrieved 2026-10-26. Source SHA256: 1a85ed47d9e38ac960f25038e22216fe2655479557779c2d052ff7b8f56be09f
MIT license reproduced in LICENSE. The loss formula and initialization are unchanged;
backward is a differentiable forward, with an observation mask for missing fields.
Huber is used as a surrogate reconstruction error, not an exact Gaussian likelihood.
