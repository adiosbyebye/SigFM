"""
A minimal TabDiffDataset that works **entirely in‑memory** and relies on
`rdt.HyperTransformer` for preprocessing, so it understands SDGym metadata.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from rdt import HyperTransformer
from rdt.transformers import (
    BinaryEncoder,
    LabelEncoder,
    GaussianNormalizer,
    OptimizedTimestampEncoder,
)


class TabDiffDataset(Dataset):
    # ------------------------------------------------------------------ init
    def __init__(self, data: pd.DataFrame, metadata: dict, device="cpu"):
        super().__init__()

        # ─── 1. Metadata ↦ sdtypes / transformer map ─────────────────────
        # tables = list(metadata["tables"].values())
        # if len(tables) != 1:
        #     raise ValueError("Only single‑table problems are supported.")
        # table_meta = tables[0]
        columns_meta = metadata["columns"]
        self.primary_key = metadata.get("primary_key")

        sdtypes, transformers = {}, {}
        cat_cols, num_cols = [], []

        for col_name, col_meta in columns_meta.items():
            if col_name == self.primary_key:
                continue

            sd = col_meta["sdtype"]
            if sd == "datetime":
                sdtypes[col_name] = "datetime"
                transformers[col_name] = OptimizedTimestampEncoder(
                    datetime_format=col_meta.get("datetime_format", "%Y-%m-%d")
                )
                num_cols.append(col_name)
            elif sd == "boolean":
                sdtypes[col_name] = "boolean"
                transformers[col_name] = BinaryEncoder()
                num_cols.append(col_name)
            elif sd == "categorical":
                sdtypes[col_name] = "categorical"
                transformers[col_name] = LabelEncoder()
                cat_cols.append(col_name)
            elif sd == "numerical":
                sdtypes[col_name] = "numerical"
                transformers[col_name] = GaussianNormalizer(
                    enforce_min_max_values=True
                )
                num_cols.append(col_name)
            elif sd == "id":
                # primary/foreign keys are ignored
                continue
            else:
                raise ValueError(f"Unknown sdtype: {sd}")

        # ─── 2. Fit + transform with HyperTransformer ───────────────────
        self.ht = HyperTransformer()
        self.ht.set_config({"sdtypes": sdtypes, "transformers": transformers})

        fit_df = (
            data.drop(columns=[self.primary_key]) if self.primary_key else data
        )
        # Keep a deterministic column order:   [num …] + [cat …]
        fit_df = fit_df[num_cols + cat_cols]

        self.transformed = self.ht.fit_transform(fit_df)
        # Ensure the transformed dataframe has the same order
        self.transformed = self.transformed[num_cols + cat_cols]

        # ─── 3. Torch tensor view ───────────────────────────────────────
        arr = self.transformed.values.astype(np.float32)
        self._torch = torch.tensor(arr, device=device)

        # ─── 4. Stats expected by TabDiff ───────────────────────────────
        self.d_numerical = len(num_cols)

        # Derive category counts safely
        tf_dict = self.ht.get_config()['transformers']      # {col_name: transformer}
        cat_sizes = []
        for col in cat_cols:
            tr = tf_dict[col]
            if isinstance(tr, LabelEncoder):
                if hasattr(tr, "categories_"):
                    cat_sizes.append(len(tr.categories_))
                elif hasattr(tr, "_categories"):
                    cat_sizes.append(len(tr._categories))
                else:   # fallback: look at the transformed data
                    cat_sizes.append(int(self.transformed[col].max()) + 1)
            else:
                # shouldn't happen but keep it safe
                cat_sizes.append(int(self.transformed[col].max()) + 1)

        self.categories = np.asarray(cat_sizes, dtype=np.int64)

    # ---------------------------------------------------------------- len / get
    def __len__(self) -> int:
        return self._torch.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._torch[idx]