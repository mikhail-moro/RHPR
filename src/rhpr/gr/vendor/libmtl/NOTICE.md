Uncertainty weighting adapted from median-research-group/LibMTL, LibMTL/weighting/UW.py.
Source: https://github.com/median-research-group/LibMTL/blob/main/LibMTL/weighting/UW.py
Retrieved 2026-10-26. Source SHA256: 1a85ed47d9e38ac960f25038e22216fe2655479557779c2d052ff7b8f56be09f
MIT license reproduced in LICENSE. The UW loss formula and initialization are unchanged;
backward is a differentiable forward, with an observation mask for missing fields.
Huber is used as a surrogate reconstruction error, not an exact Gaussian likelihood.

DWA and GradNorm in ../libmtl_balancing.py are adapted from LibMTL/weighting/DWA.py
and GradNorm.py at revision4336804847eaa5e0b924b743d76beec7ac3fdc97 (MIT).
The softmax formulas/GradNorm target match that implementation. Adaptations:
catalogue-pass observed-item averages; fixed constant-predictor normalization;
last shared decoder Linear weight for GradNorm norms (no representation proxy);
native Adagrad with a separate weight LR; incomplete-task batches skip only
the GradNorm weight update. First full pass initializes GradNorm reference;
first two passes use equal DWA weights. No second-order neural gradients.
