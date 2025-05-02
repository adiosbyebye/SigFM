# SigFM

# 🧪 SDGym Benchmarking Framework for Tabular Data Synthesizers

This repository runs a benchmarking task over multiple **tabular data synthesizers** using [SDGym](https://github.com/sdv-dev/SDGym), a framework for evaluating synthetic data models. It generates synthetic samples from real datasets and evaluates them using quality metrics like **RangeCoverage**, **KSComplement**, and **BoundaryAdherence**.

---

## 🧾 Task Description

The goal is to create a setup (conda/virtualenv), and run main.py that will conduct a series of sampling for a list of tabular data synthesizers. At the end of each run each sample will also be evaluated using specified metrics from SDGym. Intermediate results (all samples) will be saved to disk for further detailed inspection.

---

## 📦 Supported Synthesizers

Each synthesizer is preconfigured and integrated with the SDGym API (`fit()` and `sample()` methods). Here's a brief overview of each:

| Name                     | Description |
|--------------------------|-------------|
| **STaSy**                | A score-based diffusion synthesizer using SDEs and denoising score matching. |
| **CTGAN**                | Conditional Tabular GAN. Specialized in handling mixed data types using mode-specific training. |
| **TVAE**                 | Tabular Variational AutoEncoder. Models joint distribution using deep latent variables. |
| **UniformSynthesizer**   | A simple baseline that generates uniform random noise across columns. |
| **CopulaGAN**            | GAN with an explicit copula-based dependency modeling in the latent space. |
| **RealTabFormer**        | A transformer-based architecture that captures complex tabular dependencies by treating each row as sentences. |
| **CopulaSynthesizer**    | A Gaussian Copula model using user specified marginal distribution (via SDV's built-in support). |
| **TabDiff**              | A diffusion-based synthesizer designed to learn temporal and statistical tabular structure. |

---

## 📚 Available Datasets

Below is a list of tabular datasets available for benchmarking in SDGym. All datasets consist of a **single table** and vary in size and number of columns:

| Dataset Name              | Size (MB) | Columns |
|---------------------------|-----------|---------|
| `KRK_v1`                  | 0.06      | 8       |
| `adult`                   | 3.91      | 15      |
| `alarm`                   | 4.52      | 37      |
| `asia`                    | 1.28      | 8       |
| `census`                  | 98.17     | 41      |
| `census_extended`         | 4.95      | 19      |
| `child`                   | 3.20      | 20      |
| `covtype`                 | 255.65    | 55      |
| `credit`                  | 68.35     | 30      |
| `expedia_hotel_logs`      | 0.20      | 25      |
| `fake_companies`          | 0.0013    | 12      |
| `fake_hotel_guests`       | 0.03      | 9       |
| `grid`                    | 0.32      | 2       |
| `gridr`                   | 0.32      | 2       |
| `insurance`               | 3.34      | 27      |
| `intrusion`               | 162.04    | 41      |
| `mnist12`                 | 81.20     | 145     |
| `mnist28`                 | 439.60    | 785     |
| `news`                    | 18.71     | 59      |
| `ring`                    | 0.32      | 2       |
| `student_placements`      | 0.03      | 17      |
| `student_placements_pii`  | 0.03      | 18      |

To run benchmarks on one or more datasets, modify the `sdv_datasets` argument in the benchmarking call:

```python
results = sdgym.benchmark.benchmark_single_table(
    synthesizers,
    sdv_datasets=['alarm', 'adult', 'credit'],  # ← Add your desired datasets here
    ...
)
