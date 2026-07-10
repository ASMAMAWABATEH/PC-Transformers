# EXP: Cholesky Precision Weighting

## Experiment Information

Date:
2026-07-10

Branch:
feat/precision-weighting

## Base Version

Repository:
iCog-Labs-Dev/PC-Transformers

Base branch:
upstream/bench_mark

Base commit:
ad5ad7d

## Contribution Commit

Commit:
738dfb9

Message:
feat(pc): add Cholesky precision weighting to PCLayer

## Modified Files

- predictive_coding/pc_layer.py

## Implementation Summary

This experiment introduces layer-wise precision weighting for predictive coding error neurons.

### Main Changes

- Added layer-wise precision matrix storage.
- Parameterized precision matrix using Cholesky factorization:

  P = LLᵀ

- Initialized precision matrix as identity:

  L = I
  P = I

- Added precision update rule based on the free-energy gradient:

  dF/dP = 0.5S - 0.5Σ

- Updated precision after prediction error computation.
- Passed precision matrices into predictive coding inference steps.

## Motivation

Standard predictive coding assumes fixed covariance:

  Σ = I

This experiment investigates adaptive precision weighting where prediction errors are weighted according to learned uncertainty.

## Expected Effect

The model should:
- learn different uncertainty levels for different hidden layers,
- weight reliable prediction errors more strongly,
- reduce the influence of noisy error signals.

## Status

Implementation completed.

Next:
- integrate precision-weighted energy evaluation,
- run training experiments,
- compare against benchmark baseline.
