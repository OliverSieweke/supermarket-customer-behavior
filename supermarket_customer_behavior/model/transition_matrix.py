"""
Transition Matrix
=================
"""

# Data Science -------------------------------------------------------------------------
import pandas as pd

# Project ------------------------------------------------------------------------------
from supermarket_customer_behavior.data import (
    add_entrance_location,
    add_entry_exit,
    filter_non_exiting_customers,
    load_all,
)


def get_transition_matrix() -> pd.DataFrame:
    """Return locations transition matrix.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe representing the locations' transition matrix. E.g::

            next_location│ checkout    dairy    drinks   entrance   fruit     spices
            location     │
            ─────────────┼───────────────────────────────────────────────────────────
            checkout     │ 0.000000  0.000000  0.000000    1.0    0.000000  0.000000
            dairy        │ 0.392389  0.000000  0.222318    0.0    0.189852  0.195442
            drinks       │ 0.538956  0.027256  0.000000    0.0    0.217794  0.215994
            entrance     │ 0.000000  0.286639  0.153566    0.0    0.378050  0.181745
            fruit        │ 0.500784  0.236966  0.136417    0.0    0.000000  0.125833
            spices       │ 0.251672  0.323616  0.272800    0.0    0.151912  0.000000
    """
    df = load_all()
    df = add_entry_exit(df)
    df = filter_non_exiting_customers(df)
    df = add_entrance_location(df)

    df.sort_values(by=["customer_no", "timestamp"], inplace=True)
    df["next_location"] = df["location"].shift(-1).where(~df["exit"], None)

    return pd.crosstab(df["location"], df["next_location"], normalize="index")
