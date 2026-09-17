"""Workshop datasets shipped with ``ebdai``.

Tables come from
`WorkshopIgualdad2025 <https://github.com/rferper/WorkshopIgualdad2025>`_
(Raquel Fernandez Peralta and Javier Fumanal Idocin). Loaders return a
frame ready for ``train_test_split`` plus the name of the sensitive column.
"""
from __future__ import annotations

from importlib.resources import files
from typing import Optional

import numpy as np
import pandas as pd

_DATA = files('ebdai').joinpath('data')


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(_DATA.joinpath(name))


def load_titanic(prepare: bool = True) -> tuple[pd.DataFrame, Optional[str]]:
    """Titanic survival, sensitive attribute ``Sex``.

    Args:
        prepare: Drop unused columns and rows with missing age or embarkation.

    Returns:
        ``(frame, sensitive_column)``. The target column is ``Survived``.
    """
    frame = _csv('titanic.csv')
    if not prepare:
        return frame, 'Sex'
    frame = frame.dropna(subset=['Age', 'Embarked']).copy()
    frame = frame.drop(columns=['ID', 'Name', 'Ticket', 'Cabin'])
    return frame, 'Sex'


def load_heart_failure(prepare: bool = True) -> tuple[pd.DataFrame, Optional[str]]:
    """Heart-failure death, sensitive attribute ``sex`` (0 female, 1 male).

    Args:
        prepare: Cast binary clinical flags to object so they are categorical.

    Returns:
        ``(frame, sensitive_column)``. The target column is ``death``.
    """
    frame = _csv('heart_failure.csv')
    if not prepare:
        return frame, 'sex'
    frame = frame.copy()
    for column in ('anaemia', 'diabetes', 'high_blood_pressure', 'sex', 'smoking'):
        frame[column] = frame[column].astype(object)
    return frame, 'sex'


def load_loan_approval(prepare: bool = True) -> tuple[pd.DataFrame, Optional[str]]:
    """Loan approval, sensitive attribute ``Gender``.

    Args:
        prepare: Drop incomplete rows, the loan id, and recode ``Loan_Status``
            to 0/1.

    Returns:
        ``(frame, sensitive_column)``. The target column is ``Loan_Status``.
    """
    frame = _csv('loan_approval.csv')
    if not prepare:
        return frame, 'Gender'
    frame = frame.dropna().copy()
    frame = frame.drop(columns=['Loan_ID'])
    frame['Loan_Status'] = frame['Loan_Status'].map({'N': 0, 'Y': 1})
    return frame, 'Gender'


def features_and_target(
    frame: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Split a prepared frame into ``X`` and ``y``."""
    if target not in frame.columns:
        raise ValueError(f'target {target!r} is not a column of the frame')
    X = frame.drop(columns=[target])
    y = np.asarray(frame[target])
    return X, y
