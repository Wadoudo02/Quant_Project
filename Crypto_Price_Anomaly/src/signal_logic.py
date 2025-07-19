import pandas as pd
from src.config import cfg

def signal_from_zscore(
    df: pd.DataFrame,
    z_col: str = None,
    threshold: float = None,
    mode: str = None,
) -> pd.Series:
    """
    Generate a boolean entry signal based on Z-score threshold crossings.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a Z-score column.
    z_col : str, optional
        Name of the column with Z-scores. If None, defaults to:
          - 'zscore' if present, else
          - '<return_col>_zscore' based on config `return_col` or 'log_return'.
    threshold : float, optional
        Z-score threshold to trigger a signal. If None, read from
        config under `zscore.threshold` or default 2.5.
    mode : {'follow', 'fade'}, optional
        - 'follow': signal when Z ≥ threshold
        - 'fade':   signal when Z ≤ -threshold
        If None, read from config `mode` or default 'follow'.

    Returns
    -------
    pd.Series
        Boolean Series indicating signal presence at each index.
    """
    # Load configuration dict once
    cfg_dict = cfg()

    # Determine z-score column name
    if z_col is None:
        # Prefer the generic 'zscore' column added by add_zscore
        if 'zscore' in df.columns:
            z_col = 'zscore'
        else:
            return_col = cfg_dict.get('return_col', 'log_return')
            z_col = f"{return_col}_zscore"

    # Determine threshold value
    if threshold is None:
        threshold = cfg_dict.get('zscore', {}).get('threshold', 2.5)

    # Determine mode
    if mode is None:
        mode = cfg_dict.get('mode', 'follow')

    # Sanity checks
    if z_col not in df.columns:
        raise KeyError(f"DataFrame must contain column '{z_col}'")
    if mode not in ('follow', 'fade'):
        raise ValueError(f"mode must be 'follow' or 'fade', got {mode!r}")

    # Build raw boolean mask
    if mode == 'follow':
        raw_sig = df[z_col] >= threshold
    else:  # fade negative shocks
        raw_sig = df[z_col] <= -threshold

    return raw_sig.astype(bool)
