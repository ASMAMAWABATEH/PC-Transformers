# Cholesky Precision Weighting for Predictive Coding Error Neurons

## 1. Experiment Overview

### Objective

Introduce a learnable positive-definite precision matrix into the predictive coding transformer to weight prediction errors according to learned uncertainty.

The baseline predictive coding formulation assumes identity covariance:

[
\Sigma^l = I
]

which produces:

[
e^l = z^l - \mu^l
]

This experiment replaces the fixed identity precision with a learned precision matrix:

[
\Pi^l = (\Sigma^l)^{-1}
]

parameterized using a Cholesky factor:

[
\Pi^l = L^l(L^l)^T
]

where (L^l) is a learnable lower triangular matrix.

---

# 2. Baseline

## Repository State

Base implementation:

```
upstream/bench_mark
```

Base commit:

```
ad5ad7d
```

The baseline predictive coding transformer uses:

[
\Pi = I
]

meaning all prediction error dimensions have equal importance.

---

# 3. Contribution

## Main Implementation

Primary commit:

```
738dfb9 feat(pc): add Cholesky precision weighting to PCLayer
```

Additional implementation commits:

```
feat(pc): apply precision weighting to predictive coding inference

feat(config): add precision weighting configuration options

feat(config): load precision weighting hyperparameters

feat(train): enable precision weighting in training configuration
```

---

# 4. Modified Files

## predictive_coding/pc_layer.py

Purpose:

Manages layer-wise precision parameters and updates.

Changes:

* Added layer-specific precision matrix storage.
* Parameterized precision using Cholesky factor (L).
* Initialized precision as identity:

[
L = I
]

therefore:

[
\Pi = LL^T = I
]

at initialization.

* Added precision update rule based on predictive coding free-energy gradients.
* Passed learned precision matrices into predictive coding inference steps.

---

## utils/pc_utils.py

Purpose:

Applies precision weighting during predictive coding inference and energy calculation.

Changes:

Prediction error computation changed from:

[
\epsilon^l = z^l-\mu^l
]

to:

[
e^l = \Pi^l(z^l-\mu^l)
]

Implemented:

* Precision-weighted embedding errors.
* Precision-weighted hidden-layer errors.
* Precision-weighted attention errors.
* Precision-aware NLL energy calculation.

The NLL energy formulation:

[
F =
\frac{1}{2}(z-\mu)^T\Pi(z-\mu)
-\frac{1}{2}\log|\Pi|
]

---

## predictive_coding/config.py

Added precision configuration parameters:

```python
use_precision_weighting
precision_lr
```

Purpose:

* Enable or disable precision weighting.
* Control precision matrix learning rate.

---

## utils/config_utils.py

Added configuration support for:

```python
use_precision_weighting
precision_lr
```

including fallback values when configuration files do not contain these parameters.

---

## training.py

Connected precision configuration into model initialization:

```python
use_precision_weighting=best_config["use_precision_weighting"]

precision_lr=best_config["precision_lr"]
```

This enables training runs to use the Cholesky precision mechanism.

---

# 5. Mathematical Formulation

For each predictive coding layer:

Prediction error:

[
\epsilon^l=z^l-\mu^l
]

Precision-weighted error neuron:

[
e^l=\Pi^l\epsilon^l
]

Cholesky parameterization:

[
\Pi^l=L^l(L^l)^T
]

This guarantees a valid positive-definite precision matrix:

[
\Pi^l \succ 0
]

and avoids unstable unconstrained matrix learning.

---

# 6. Motivation

The original predictive coding transformer assumes every prediction error dimension has identical uncertainty.

Using:

[
\Sigma=I
]

means every error component contributes equally during inference.

The Cholesky precision formulation allows the model to learn:

* reliable dimensions,
* uncertain dimensions,
* adaptive error weighting.

The model can therefore modify how strongly different prediction errors influence hidden-state updates.

---

# 7. Expected Impact

The hypothesis of this experiment is that learned precision weighting can:

* improve stability of predictive coding inference,
* provide uncertainty-aware error propagation,
* reduce the influence of noisy prediction errors,
* improve optimization of predictive coding energy.

---

## 8. Results

The precision weighting implementation was successfully integrated and trained for 5 epochs. The final training run completed without errors, producing the final checkpoint and evaluation metrics.

### Final Training Metrics

```
Epoch: 5/5

Train Energy: 6754.6976
Train Perplexity: 2.8889

Validation Energy: 7197.5706
Validation Perplexity: 2.8213
```

### Training Completion

```
Training completed in 6727.34 seconds

Saved checkpoint:
checkpoints/model_epoch_5.pt

Final model:
checkpoints/final_model.pt
```

### Observation

The experiment achieved a validation perplexity of **2.8213**, showing that the model maintained stable language modeling performance after introducing precision weighting. The final energy values indicate that the precision-weighted predictive coding energy formulation was successfully optimized during training.

The experiment confirms that the new precision weighting configuration can be integrated into the predictive coding transformer pipeline without breaking training stability or model checkpoint generation.

