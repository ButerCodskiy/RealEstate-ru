import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from src.features.target_encoding import OOFTargetEncoder


def test_oof_target_encoder_no_leakage():
    """
    Unit test to verify that OOFTargetEncoder does NOT leak the sample's own target.
    If sample i's target is modified, its OOF encoded value MUST remain identical.
    """
    df_toy = pd.DataFrame({
        "Метро_1": ["Арбатская", "Арбатская", "Арбатская", "Тверская", "Тверская", "Тверская", "Киевская", "Киевская", "Киевская", "Киевская"],
        "log_Цена": [16.0, 16.5, 17.0, 15.0, 15.5, 16.0, 14.0, 14.5, 15.0, 15.5]
    })

    # Fixed folds: fold 0 is indices [0, 1, 2, 3, 4], fold 1 is indices [5, 6, 7, 8, 9]
    folds = [(np.array([5, 6, 7, 8, 9]), np.array([0, 1, 2, 3, 4])),
             (np.array([0, 1, 2, 3, 4]), np.array([5, 6, 7, 8, 9]))]

    encoder = OOFTargetEncoder(cols=["Метро_1"], target_col="log_Цена", alpha=1.0, min_samples_leaf=1)
    res1 = encoder.fit_transform_oof(df_toy, folds=folds)
    val_sample_0_orig = res1["Метро_1_te"].iloc[0]

    # Now modify sample 0's target by a huge amount (+100.0)
    df_toy_perturbed = df_toy.copy()
    df_toy_perturbed.loc[0, "log_Цена"] = 116.0

    encoder2 = OOFTargetEncoder(cols=["Метро_1"], target_col="log_Цена", alpha=1.0, min_samples_leaf=1)
    res2 = encoder2.fit_transform_oof(df_toy_perturbed, folds=folds)
    val_sample_0_perturbed = res2["Метро_1_te"].iloc[0]

    assert np.isclose(val_sample_0_orig, val_sample_0_perturbed), (
        f"LEAKAGE DETECTED! Encoded value changed from {val_sample_0_orig} to {val_sample_0_perturbed} when sample's own target changed!"
    )
