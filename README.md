# Mathematics for Machine Learning
---
Author: Ameeya Bhusan Sahoo

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![Tests](https://img.shields.io/badge/tests-pytest-0A9EDC?logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

This repository works through the core mathematics of ML - Linear Algebra, calculus, probability, optimization, and then turns that mathematics into working models (linear regression, logistic regression, PCA). Every algorithm is implemented so the formula is visible in the code, documented in a notebook, and covered by tests that check it against NumPy or scikit-learn.
---
## Why the maths matters?

It is easy to call `model.fit(X, y)` and get a number back. It is much more
useful to know *what that number is, why the optimiser converged, and when the
method will break*. Understanding the underlying mathematics is what separates
running a model from reasoning about one, diagnosing a divergent training run,
choosing a sensible regulariser, or explaining a result to a stakeholder all
depend on it. This repo is my attempt to make that understanding concrete by
rebuilding the foundations rather than importing them.

## What this repository demonstrates

- **Mathematical maturity** — derivations rendered in LaTeX, connected directly
  to code.
- **Clean, tested Python** — small modules with a consistent API and a passing
  `pytest` suite.
- **From-scratch implementations** — NumPy only; scikit-learn is used as a
  baseline to validate against, never to hide the work.
- **Clear communication** — notebooks that explain, visualise, and summarise
  each idea.

## Repository structure

```
math-for-ml-from-scratch/
│
├── README.md
├── requirements.txt
├── environment.yml
├── LICENSE
├── .gitignore
│
├── notebooks/                 # one notebook per topic: theory + code + plots
│   ├── 01_linear_algebra_vectors_matrices.ipynb
│   ├── 02_eigenvalues_pca.ipynb
│   ├── 03_calculus_gradients_optimization.ipynb
│   ├── 04_probability_statistics.ipynb
│   ├── 05_linear_regression_from_scratch.ipynb
│   ├── 06_logistic_regression_from_scratch.ipynb
│   ├── 07_gradient_descent_visualization.ipynb
│   └── 08_pca_dimensionality_reduction.ipynb
│
├── src/                       # the library: clean, importable implementations
│   ├── __init__.py
│   ├── linear_algebra.py
│   ├── calculus.py
│   ├── probability.py
│   ├── optimization.py
│   ├── regression.py
│   └── visualization.py
│
├── tests/                     # pytest suite validating the library
│   ├── test_linear_algebra.py
│   ├── test_probability.py
│   ├── test_optimization.py
│   └── test_regression.py
│
├── examples/                  # short runnable scripts
│   ├── linear_regression_demo.py
│   ├── logistic_regression_demo.py
│   └── pca_demo.py
│
├── docs/                      # notes, interview prep, roadmap, references
│   ├── math_notes.md
│   ├── interview_questions.md
│   ├── learning_roadmap.md
│   └── references.md
│
└── assets/figures/            # rendered example figures
```

## Topics covered

| Area | Concepts | Code |
|------|----------|------|
| Linear algebra | dot product, norms, cosine similarity, matrix multiplication, eigenvectors, power iteration | `src/linear_algebra.py` |
| Calculus | numerical derivatives, gradients, Jacobians | `src/calculus.py` |
| Probability | mean, variance, covariance, correlation, Gaussian/Bernoulli/Binomial | `src/probability.py` |
| Optimisation | gradient descent, momentum, trajectory tracking | `src/optimization.py` |
| Models | linear regression, logistic regression, PCA | `src/regression.py` |

## How to run

Clone the repo, set up the environment, and run the tests.

```bash
git clone https://github.com/<your-username>/math-for-ml-from-scratch.git
cd math-for-ml-from-scratch

# Option A: pip
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate math-for-ml

# run the test suite
pytest -q

# run a demo
python examples/linear_regression_demo.py

# launch the notebooks
jupyter notebook
```

## Example outputs

| Linear regression | Gradient descent | PCA projection |
|---|---|---|
| ![regression](assets/figures/linear_regression.png) | ![gd](assets/figures/gradient_descent.png) | ![pca](assets/figures/pca_projection.png) |

```text
$ python examples/pca_demo.py
Explained variance ratio per component:
  PC1: 0.9848
  PC2: 0.0082
  PC3: 0.0070
First component alone explains 98.5% of the variance.
```

## Skills demonstrated

- Translating mathematical definitions into correct, readable NumPy.
- Writing a small, consistent library API (`fit` / `predict`, returned result
  objects) and a meaningful `pytest` suite.
- Numerical reasoning: finite-difference gradient checks, conditioning,
  numerically stable sigmoid and least-squares solves.
- Technical communication through documented notebooks and clean figures.

## How this connects to my background

I come from theoretical and computational chemistry, electronic-structure
methods, quantum simulation, and large-scale Python scientific computing. A
surprising amount of that work *is* this mathematics: diagonalising symmetric
operators, checking gradients with finite differences, and reasoning in
orthogonal bases. This repository is where I make the transfer explicit and
bridge toward applied ML and data science:

- **Data science** — the regression, classification, and dimensionality-reduction
  workflows here are the everyday tools of the field.
- **Scientific machine learning** — gradient-based optimisation and linear-algebra
  fluency carry directly into physics-informed and surrogate models.
- **Computational materials / quantum chemistry** — PCA is the same covariance
  diagonalisation used to analyse high-dimensional simulation data.
- **Model interpretability** — understanding the maths is the prerequisite for
  explaining *why* a model behaves as it does, not just *that* it works.

## Future roadmap

Planned additions include Ridge/Lasso regularisation, SVD, k-means, additional
optimisers (Adam, RMSProp), and a small from-scratch neural network. See
[`docs/learning_roadmap.md`](docs/learning_roadmap.md) for the full plan.

## References

Key sources are listed in [`docs/references.md`](docs/references.md). The
implementations are validated against NumPy and scikit-learn; all explanations
and code are my own.

## About me

I am a computational scientist with a background in theoretical/computational
chemistry, density functional theory, quantum simulation, and Python scientific
computing, now moving toward data science and applied machine learning. I learn
best by building things end to end and by connecting abstract mathematics to
concrete implementation, which is exactly what this repository is. I am
especially interested in scientific ML and in roles where mathematical rigour
and clean engineering both matter.

- **Email:** `ameeyabhusansahoo5@gmail.com`

## License

Released under the MIT License — see [`LICENSE`](LICENSE).