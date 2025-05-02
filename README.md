# SigFM

# 🧪 SDGym Benchmarking Framework for Tabular Data Synthesizers

This repository runs a benchmarking task over multiple **tabular data synthesizers** using [SDGym](https://github.com/sdv-dev/SDGym), a framework for evaluating synthetic data models. It generates synthetic samples from real datasets and evaluates them using quality metrics like **RangeCoverage**, **KSComplement**, and **BoundaryAdherence**.

---

## 🧾 Task Description

You will train **multiple synthesizer models** (generative models for tabular data), **generate synthetic datasets**, and then **evaluate them** using metrics from SDGym. Results will be automatically saved to disk for inspection.

---

## 📦 Supported Synthesizers

Each synthesizer is preconfigured and integrated with the SDGym API (`fit()` and `sample()` methods). Here's a brief overview of each:

| Name                     | Description |
|--------------------------|-------------|
| **STaSy**                | A score-based diffusion synthesizer using SDEs and denoising score matching. Requires careful preprocessing via `RDT`. |
| **CTGAN**                | Conditional Tabular GAN. Specialized in handling mixed data types using mode-specific training. |
| **TVAE**                 | Tabular Variational AutoEncoder. Models joint distribution using deep latent variables. |
| **UniformSynthesizer**   | A simple baseline that generates uniform random noise across columns. |
| **CopulaGAN**            | GAN with an explicit copula-based dependency modeling in the latent space. |
| **RealTabFormer**        | A transformer-based architecture that captures complex tabular dependencies. |
| **CopulaSynthesizer**    | A Gaussian Copula model using truncated normal marginals (via SDV's built-in support). |
| **TabDiff**              | A diffusion-based synthesizer designed to learn temporal and statistical tabular structure. |

---

## 🧑‍💻 How To Run

To launch a benchmark experiment on a dataset (e.g., `KRK_v1`):
