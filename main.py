import math
import os
import matplotlib
matplotlib.use('Agg')  # Set matplotlib backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils import tensorboard
from torch.utils.data import DataLoader, TensorDataset
from ml_collections import config_dict, config_flags


import sdgym
from sdgym import create_sdv_synthesizer_variant
from sdgym.synthesizers import (
    CTGANSynthesizer,
    TVAESynthesizer,
    UniformSynthesizer,
    CopulaGANSynthesizer,
    RealTabFormerSynthesizer,
)


# RDT Transformers, for encoding of non-numerical columns
from rdt import HyperTransformer
from rdt.transformers import (
    OptimizedTimestampEncoder,
    BinaryEncoder,
    LabelEncoder,
    FloatFormatter,
    GaussianNormalizer
)

# =================================================================
# STaSy-related imports
# =================================================================
import STaSy.datasets as datasets
import STaSy.evaluation
import STaSy.likelihood as likelihood
import STaSy.losses as losses
import STaSy.models.ncsnpp_tabular
import STaSy.sampling as sampling_
import STaSy.sde_lib as sde_lib
from STaSy.models import utils as mutils
from STaSy.models.ema import ExponentialMovingAverage
from STaSy.utils import save_checkpoint, restore_checkpoint, apply_activate

_original_DataParallel = nn.DataParallel

class NoParallelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

# Patch DataParallel to avoid using it, but still be a subclass of nn.Module
nn.DataParallel = NoParallelWrapper

class STaSySynethesizer:
    def __init__(self, config):
        """
        Initialize the STaSy trainer with all needed components:
            - Configuration
            - Model creation
            - SDE definition
            - Optimizer & EMA
            - Misc. utilities (loss function, sampling function, etc.)
        """
        self.config = config
        self.device = self.config.device

        # Create the score model
        self.score_model = mutils.create_model(self.config).to(self.device)

        # Define the SDE
        self.sde, self.sampling_eps = self._create_sde(self.config.training.sde)

        # TensorBoard writer
        self.tb_dir = os.path.join(self.config.workdir, "tensorboard")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = tensorboard.SummaryWriter(self.tb_dir)

        # Exponential Moving Average
        self.ema = ExponentialMovingAverage(
            self.score_model.parameters(),
            decay=self.config.model.ema_rate
        )

        # Optimizer
        self.optimizer = losses.get_optimizer(
            self.config, self.score_model.parameters()
        )

        # State dictionary
        self.state = dict(
            optimizer=self.optimizer,
            model=self.score_model,
            ema=self.ema,
            step=0,
            epoch=0
        )

        # Checkpoint directories
        self.checkpoint_dir = os.path.join(self.config.workdir, "checkpoints")
        self.checkpoint_meta_dir = os.path.join(
            self.config.workdir, "checkpoints-meta", "checkpoint.pth"
        )
        self.checkpoint_finetune_dir = os.path.join(
            self.config.workdir, "checkpoints_finetune"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.checkpoint_meta_dir), exist_ok=True)
        os.makedirs(self.checkpoint_finetune_dir, exist_ok=True)

        # We will define these later in fit(), once we know the data shape
        self.train_iter = None
        self.transformer = None
        self.meta = None

        # Data scalers (optional; define or remove as needed)
        self.scaler = None
        self.inverse_scaler = None

        # SPL parameters
        self.alpha0 = self.config.model.alpha0
        self.beta0 = self.config.model.beta0

        # Additional config for sampling
        self.sampling_fn = None

        print(f"Score Model has {sum(p.numel() for p in self.score_model.parameters())} parameters.")

    def fit(self, data, metadata):
        """
        Trains the model using transformed data via RDT HyperTransformer.
        Args:
            data (pd.DataFrame): Original real data
            metadata (dict): Column metadata (SINGLE_TABLE_V1 format)
        """

        # Store primary key for use in sample()
        self.primary_key = metadata.get('primary_key')
        sdtypes = {}
        transformers = {}

        # Define RDT HyperTransformer
        # Inspect each column in metadata
        for col_name, col_meta in metadata['columns'].items():
            print("col_name is: ",col_name)
            print("col_meta is: ",col_meta)
            col_sdtype = col_meta['sdtype']

            # Skip modeling the primary key
            # (We will add it back after sampling)
            if col_name == self.primary_key:
                continue

            if col_sdtype == 'datetime':
                sdtypes[col_name] = 'datetime'
                dt_format = col_meta.get('datetime_format', '%Y-%m-%d')
                transformers[col_name] = OptimizedTimestampEncoder(datetime_format=dt_format)
            elif col_sdtype == 'boolean':
                sdtypes[col_name] = 'boolean'
                # Let RDT pick the default BooleanTransformer
                transformers[col_name] = BinaryEncoder()
            elif col_sdtype == 'categorical':
                sdtypes[col_name] = 'categorical'
                # Let RDT pick the default CategoricalTransformer
                transformers[col_name] = LabelEncoder()
            elif col_sdtype == 'numerical':
                sdtypes[col_name] = 'numerical'
                # Let RDT pick the default NumericTransformer
                transformers[col_name] = GaussianNormalizer(enforce_min_max_values=True)
            elif col_sdtype == 'id':
                # If you want to skip IDs from modeling, you can just skip them altogether.
                # Alternatively, treat them as 'categorical'.
                # For now, let's skip them from modeling just like the primary_key.
                continue
            else:
                raise ValueError(f"Unknown SdType: {col_sdtype}")
        column_transforms = {
            'sdtypes': sdtypes,
            'transformers': transformers
        }
        # Initialize and fit the transformer
        self.ht = HyperTransformer()
        self.ht.set_config(column_transforms)
        transformed_data = self.ht.fit_transform(data.drop(columns=[self.primary_key]) if self.primary_key else data)

        # Convert to torch.Tensor
        numpy_data = transformed_data.values.astype(np.float32)
        tensor_data = torch.tensor(numpy_data, dtype=torch.float32, device=self.device)
        self.data_dim = tensor_data.shape[1]

        # DataLoader
        train_ds = TensorDataset(tensor_data)
        self.train_iter = DataLoader(
            train_ds, batch_size=self.config.training.batch_size, shuffle=True
        )

        self.optimize_fn = losses.optimization_manager(self.config)
        sampling_shape = (self.config.training.batch_size, self.data_dim)
        self.sampling_fn = sampling_.get_sampling_fn(
            self.config, self.sde, sampling_shape, None, self.sampling_eps
        )

        # ----------------------------------------------------------------
        # 2) Main Training Loop
        # ----------------------------------------------------------------
        for epoch in range(self.config.training.epoch + 1):
            self.state['epoch'] += 1
            for iteration, batch_data in enumerate(self.train_iter):
                batch = batch_data[0].to(self.device).float()  # (B, D)
                self.optimizer.zero_grad()

                # Compute per-sample loss
                loss_values = self._loss_fn(self.score_model, batch)

                # SPL alpha/beta quantiles
                q_alpha_val = (
                    self.alpha0 + torch.log(
                        torch.tensor(
                            1 + 0.01718 * self.state['step'] * (1 - self.alpha0),
                            dtype=torch.float32
                        )
                    )
                ).clamp_(max=1).to(loss_values.device)

                q_beta_val = (
                    self.beta0 + torch.log(
                        torch.tensor(
                            1 + 0.01718 * self.state['step'] * (1 - self.beta0),
                            dtype=torch.float32
                        )
                    )
                ).clamp_(max=1).to(loss_values.device)

                alpha = torch.quantile(loss_values, q_alpha_val)
                beta = torch.quantile(loss_values, q_beta_val)
                assert alpha <= beta

                # Compute SPL weights
                v = self._compute_v(loss_values, alpha, beta)
                loss = torch.mean(v * loss_values)

                # Backprop and update
                loss.backward()
                self.optimize_fn(self.optimizer, self.score_model.parameters(), step=self.state['step'])
                self.state['step'] += 1
                self.state['ema'].update(self.score_model.parameters())

            # Logging and periodic checkpoint
            print(f"epoch: {epoch}, iter: {iteration}, training_loss: {loss.item():.5e}, "
                  f"q_alpha: {q_alpha_val:.3e}, q_beta: {q_beta_val:.3e}")

            if epoch % 10 == 0:
                save_checkpoint(self.checkpoint_meta_dir, self.state)



    def sample(self, synthesizer, n_rows=100):
        """
        Generates synthetic data and inverts transformation using RDT.
        Args:
            synthesizer (STaSySynethesizer): Trained instance
            n_rows (int): Number of rows to sample
        Returns:
            pd.DataFrame: Synthetic samples in original schema
        """
        sampling_shape = (n_rows, self.data_dim)
        samples, _ = self.sampling_fn(self.score_model, sampling_shape=sampling_shape)
        samples = samples.detach().cpu().numpy()
        synthetic_df = pd.DataFrame(samples, columns=self.ht._output_columns)

        # Inverse transform back to original format
        synthetic_df = self.ht.reverse_transform(synthetic_df)

        # Add dummy or randomly sampled primary keys if needed
        if self.primary_key:
            synthetic_df[self.primary_key] = [f'pk_{i}' for i in range(n_rows)]

        return synthetic_df

    # -------------------------------------------------------------------------
    # Internal / utility methods below
    # -------------------------------------------------------------------------

    def _create_sde(self, sde_type):
        """
        Build SDE object from config.
        """
        if sde_type.lower() == 'vpsde':
            sde_obj = sde_lib.VPSDE(
                beta_min=self.config.model.beta_min,
                beta_max=self.config.model.beta_max,
                N=self.config.model.num_scales
            )
            sampling_eps = 1e-3
        elif sde_type.lower() == 'subvpsde':
            sde_obj = sde_lib.subVPSDE(
                beta_min=self.config.model.beta_min,
                beta_max=self.config.model.beta_max,
                N=self.config.model.num_scales
            )
            sampling_eps = 1e-3
        elif sde_type.lower() == 'vesde':
            sde_obj = sde_lib.VESDE(
                sigma_min=self.config.model.sigma_min,
                sigma_max=self.config.model.sigma_max,
                N=self.config.model.num_scales
            )
            sampling_eps = 1e-5
        else:
            raise NotImplementedError(f"SDE {sde_type} unknown.")
        return sde_obj, sampling_eps

    def _loss_fn(self, model, batch):
        """
        Denoising Score Matching loss.
        Returns a per-sample loss tensor of shape (batch_size,).
        """
        continuous = self.config.training.continuous
        score_fn = mutils.get_score_fn(
            self.sde, model, train=True, continuous=continuous
        )
        # t ~ Uniform(0, T)
        t = torch.rand(batch.shape[0], device=batch.device) * (self.sde.T - 1e-5) + 1e-5
        z = torch.randn_like(batch)
        mean, std = self.sde.marginal_prob(batch, t)
        perturbed_data = mean + std[:, None] * z
        score = score_fn(perturbed_data, t)

        # Loss
        loss_values = torch.square(score * std[:, None] + z)
        loss_values = torch.mean(loss_values.reshape(loss_values.shape[0], -1), dim=-1)
        return loss_values

    def _min_max_scaling(self, factor, scale=(0, 1)):
        """
        Scale values in factor to [scale[0], scale[1]] range.
        """
        eps = 1e-12
        std = (factor - factor.min()) / (factor.max() - factor.min() + eps)
        new_min, new_max = scale
        return std * (new_max - new_min) + new_min

    def _compute_v(self, ll, alpha, beta):
        """
        Compute self-paced weights v in [0,1]:
          - Hard samples (loss > beta) => v=0,
          - Easy samples (loss <= alpha) => v=1,
          - In-between => linearly scale from 1 to 0.
        """
        v = -torch.ones_like(ll).to(ll.device)
        v[torch.gt(ll, beta)] = 0.
        v[torch.le(ll, alpha)] = 1.

        # For in-between samples, scale from 1 to 0
        mask = (v == -1)
        in_between = ll[mask]
        if len(in_between) > 1:
            v[mask] = self._min_max_scaling(in_between, scale=(1, 0)).to(v.device)
        else:
            # If there's only 1 sample (or none) in-between, just set it to 0.5
            v[mask] = 0.5

        return v


if __name__ == "__main__":
    config = config_dict.ConfigDict()
    config.workdir = "stasy"
    config.seed = 42
    config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # Training configuration
    config.training = training = config_dict.ConfigDict()
    training.batch_size = 1000
    training.epoch = 100
    training.likelihood_weighting = False
    training.continuous = True
    training.reduce_mean = False
    training.eps = 1e-05
    training.loss_weighting = False
    training.spl = False
    training.lambda_ = 0.5
    training.sde = 'vesde'
    training.n_iters = 100000
    training.tolerance = 1e-03
    training.hutchinson_type = "Rademacher"
    training.retrain_type = "median"
    training.eps_iters = 1
    training.fine_tune_epochs = 1

    # Sampling configuration
    config.sampling = sampling = config_dict.ConfigDict()
    sampling.n_steps_each = 1
    sampling.noise_removal = False
    sampling.probability_flow = True
    sampling.snr = 0.16
    sampling.method = 'ode'
    sampling.predictor = 'euler_maruyama'
    sampling.corrector = 'none'

    # Data configuration
    config.data = data = config_dict.ConfigDict()
    data.centered = False
    data.uniform_dequantization = False
    data.image_size = 7  #Number of Columns, 37 for alarm 7 for KRK_v1

    # Model configuration
    config.model = model = config_dict.ConfigDict()
    model.nf = 64
    model.hidden_dims = (64, 128, 256, 128, 64)
    model.conditional = True
    model.embedding_type = 'fourier'
    model.fourier_scale = 16
    model.layer_type = 'concatsquash'
    model.name = 'ncsnpp_tabular'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.activation = 'elu'
    model.sigma_min = 0.01
    model.sigma_max = 10.
    model.num_scales = 50
    model.alpha0 = 0.3
    model.beta0 = 0.95

    # Optimizer configuration
    config.optim = optim = config_dict.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'Adam'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 5000
    optim.grad_clip = 1.

    # Instantiate trainer
    trainer = STaSySynethesizer(config)
    stasy_synth = sdgym.create_single_table_synthesizer(
        get_trained_synthesizer_fn=trainer.fit,
        sample_from_synthesizer_fn=trainer.sample,
        display_name="STaSy"
    )

    # =================================================================
    # CopulaSynthesizer-related imports
    # =================================================================
    ctgan_synth = CTGANSynthesizer()
    tvae_synth = TVAESynthesizer()
    uniform_synth = UniformSynthesizer()
    cop_gan_synth = CopulaGANSynthesizer()
    realTF_synth = RealTabFormerSynthesizer()
    CopulaSynthesizer = create_sdv_synthesizer_variant(
        synthesizer_class='GaussianCopulaSynthesizer',
        synthesizer_parameters={'default_distribution': 'truncnorm'},
        # Possible options include the following:
        # 'norm': copulas.univariate.GaussianUnivariate,
        # 'beta': copulas.univariate.BetaUnivariate,
        # 'truncnorm': copulas.univariate.TruncatedGaussian,
        # 'gamma': copulas.univariate.GammaUnivariate,
        # 'uniform': copulas.univariate.UniformUnivariate,
        # 'gaussian_kde': copulas.univariate.GaussianKDE,
        display_name = 'TruncNormCopulaSynthesizer'
    )

    # =================================================================
    # TabDiff (Diffusion)-related imports
    # =================================================================

    # 1. Preprocess and build dataset/objects
    from tabdiff.tabdiff.modules.main_modules import UniModMLP, Model
    from tabdiff.tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
    from tabdiff.tabdiff.trainer import Trainer  # original trainer.py
    from tabdiff.tabdiff_dataset_ht import TabDiffDataset

    class _DummyLogger:
        def define_metric(self, *a, **kw):  ...

        def log(self, *a, **kw):            ...
    class _DummyMetrics:
        """Just enough surface for Trainer; no heavy evaluation during SDGym fit()."""

        def __init__(self, n_rows: int):
            self.real_data_size = n_rows  # used when Trainer wants default
            self.info = {'task_type': 'unsupervised'}
    # ────────────────────────────────────────────────────────────────────────────────
    class TabDiffSynthesizer:
        """
        A very thin adapter: wraps TabDiff’s diffusion model so that SDGym only sees
        `fit(data, metadata)` and `sample(n)`.

        All expensive evaluations inside the original Trainer are disabled by
        setting `check_val_every` *larger* than the number of training epochs.
        """

        # --------------------------------------------------------------------- init
        def __init__(
                self,
                hidden_dim: int = 64,
                learning_rate: float = 1e-3,
                epochs: int = 50,
                batch_size: int = 256,
                weight_decay: float = 1e-5,
                device: str | torch.device | None = None,
        ) -> None:
            self.hidden_dim = hidden_dim
            self.learning_rate = learning_rate
            self.epochs = epochs
            self.batch_size = batch_size
            self.weight_decay = weight_decay
            self.device = "cpu"
            # self.device = (
            #     torch.device(device)
            #     if device is not None
            #     else torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # )

            self._dataset: TabDiffDataset | None = None
            self._trainer: Trainer | None = None
            self._ht = None  # HyperTransformer (for inverse transform)
            self._trained = False

        # --------------------------------------------------------------------- fit
        def fit(self, data: pd.DataFrame, metadata: dict) -> None:  # SDGym API
            """
            Transform *data* ↦ numerical tensor, build TabDiff diffusion, and train.

            Parameters
            ----------
            data      : raw table as supplied by SDGym
            metadata  : single‑table metadata dict (sdtypes, etc.)
            """
            # 1. build dataset (HyperTransformer lives *inside* it)
            self._dataset = TabDiffDataset(data, metadata, device=self.device)
            self._ht = self._dataset.ht  # keep for inverse later
            train_loader = DataLoader(
                self._dataset, batch_size=self.batch_size, shuffle=True
            )

            # 2. instantiate TabDiff denoiser backbone
            unimod_cfg = dict(
                d_numerical=self._dataset.d_numerical,
                categories=(self._dataset.categories + 1).tolist(),
                num_layers=5,
                d_token=64
                if len(self._dataset.categories) > 0 else [1],
                dim_t=self.hidden_dim,
                use_mlp=True,
            )
            backbone = UniModMLP(**unimod_cfg)
            denoise = Model(backbone, sigma_data=1.0, sigma_min=0.002,
                            sigma_max=80., rho=7.).to(self.device)

            # 3. diffusion wrapper
            diffusion = UnifiedCtimeDiffusion(
                num_classes=self._dataset.categories,
                num_numerical_features=self._dataset.d_numerical,
                denoise_fn=denoise,
                y_only_model=None,
                scheduler="power_mean_per_column",
                cat_scheduler="log_linear_per_column",
                num_timesteps=200,
                noise_dist="uniform_t",
                edm_params={"sigma_data": 1.0},
                sampler_params={
                    "stochastic_sampler": False,  # simplest deterministic sampler
                    "second_order_correction": False,
                },
                device=self.device,
            )

            # 4. trainer (evaluation disabled)
            self._trainer = Trainer(
                diffusion=diffusion,
                train_iter=train_loader,
                dataset=self._dataset,
                test_dataset=self._dataset,  # placeholder
                metrics=_DummyMetrics(len(data)),
                logger=_DummyLogger(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                steps=self.epochs,
                batch_size=self.batch_size,
                check_val_every=self.epochs + 1,  # <- never triggered
                sample_batch_size=self.batch_size,
                model_save_path="./_tabdiff_ckpt",  # temp local dir
                result_save_path="./_tabdiff_results",
                device=self.device,
            )
            self._trainer.run_loop()
            self._trained = True

        # ------------------------------------------------------------------ sample
        @torch.no_grad()
        def sample(self, synthesizer, n_rows: int) -> pd.DataFrame:  # SDGym API
            """
            Draw *n* synthetic rows and reverse‑transform them back to the original
            schema using the stored HyperTransformer.
            """
            if not self._trained:
                raise RuntimeError("Call `.fit()` before `.sample()`")

            # 1. raw tensor from TabDiff model  (shape: [n, d_total])
            synth_tensor = self._trainer.diffusion.sample_all(
                n_rows, self.batch_size, keep_nan_samples=False
            ).cpu().numpy()

            # 2. build DataFrame with *transformed* column names (same order)
            transformed_cols = self._dataset.transformed.columns.tolist()
            df_trans = pd.DataFrame(synth_tensor, columns=transformed_cols)

            # 3. inverse transform back to user space
            df_orig = self._ht.reverse_transform(df_trans)

            # 4. re‑insert primary key if it existed (simple incremental ids)
            if self._dataset.primary_key is not None:
                df_orig[self._dataset.primary_key] = (
                    np.arange(len(df_orig)).astype(int)
                )

            return df_orig.reset_index(drop=True)

    TabDiffSynth = TabDiffSynthesizer(
        hidden_dim=64,
        learning_rate=1e-3,
        epochs=100,
        batch_size=256,
    )

    tabdiff_synth = sdgym.create_single_table_synthesizer(
        get_trained_synthesizer_fn=TabDiffSynth.fit,
        sample_from_synthesizer_fn=TabDiffSynth.sample,
        display_name="TabDiff Copula Synthesizer"
    )



    # =================================================================
    # Testing Framework
    # =================================================================
    synthesizers = [stasy_synth, ctgan_synth, tvae_synth, uniform_synth, cop_gan_synth, realTF_synth, CopulaSynthesizer]

    results = sdgym.benchmark.benchmark_single_table(
        synthesizers,
        sdv_datasets=['KRK_v1'], #'KRK_v1', 'alarm', 'intrusion', 'insurance', 'adult'
        show_progress=True,
        sdmetrics= [
            'RangeCoverage',
            'BoundaryAdherence',
            'KSComplement',
            # 'DCRBaselineProtection'
        ],
        output_filepath='./results/benchmarking_output_1.csv',
        detailed_results_folder='./results/results_detailed_1',
    )
    print("done\n")
    print("SDGym Benchmark Results:", results)



