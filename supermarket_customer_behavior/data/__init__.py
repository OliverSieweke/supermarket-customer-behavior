"""
Data
====

This module provides utility methods for manipulating the original data.
"""

# Standard Library ---------------------------------------------------------------------
import collections
from typing import Iterable, Union

# Data Science -------------------------------------------------------------------------
import pandas as pd

# Project ------------------------------------------------------------------------------
from supermarket_customer_behavior.paths import WeekDay, day_data_file_path


def load_day(day: WeekDay, prefix_customer_no: bool = False) -> pd.DataFrame:
    """Return dataframe containing the data of the specified day.

    Parameters
    ----------
    day
        Week day.

    prefix_customer_no
        If ``True``, prefixes the customer number with the day of the
        week. (Useful when combining data from several days)

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing the data for the given day. E.g::

                timestamp            customer_no  location
                -------------------  -----------  --------
            0   2019-09-02 07:03:00  1            dairy
            1   2019-09-02 07:03:00  2            dairy
    """
    df = pd.read_csv(day_data_file_path(day), sep=";", parse_dates=["timestamp"])

    if prefix_customer_no:
        df["customer_no"] = day.value + "_" + df["customer_no"].astype(str)

    return df


def load_all() -> pd.DataFrame:
    """Return dataframe containing the data from all days.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing the data from all days. E.g::

            timestamp            customer_no  location
            -------------------  -----------  --------
            2019-09-02 07:03:00     monday_1     dairy
            2019-09-06 21:50:00  friday_1500     dairy
    """
    return pd.concat(
        [load_day(day, prefix_customer_no=True) for day in WeekDay], axis="index"
    ).reset_index(drop=True)


def add_entry_exit(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns specifying if a customer is entering or exiting the store.

    Notes
    -----
    Some customers don't reach the checkout before the store closes.
    Those customers can be filtered out with
    :func:`filter_non_exiting_customers`.

    Parameters
    ----------
    df
        Dataframe containing at least the ``timestamp`` and
        ``customer_no`` columns.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe augmented with the ``entry``, ``exit`` and
        ``customer_count_change`` columns. E.g::

                timestamp            customer_no  location  entry  exit   customer_count_change
                -------------------  -----------  --------  -----  -----  ---------------------
            0   2019-09-02 07:03:00  1            dairy     True   False                      1
            5   2019-09-02 07:04:00  6            spices    True   False                      1
            8   2019-09-02 07:05:00  1            checkout  False  True                      -1
            10  2019-09-02 07:05:00  6            dairy     False  False                      0
            17  2019-09-02 07:06:00  12           spices    True   False                      1
            21  2019-09-02 07:07:00  12           drinks    False  False                      0
            31  2019-09-02 07:10:00  12           checkout  False  True                      -1
    """
    df.sort_values(by="timestamp", inplace=True)
    # A customer is considered to be entering the store the first time
    # he appears in the data:
    df["entry"] = ~df["customer_no"].duplicated()
    # A customer is considered to be exiting the store when he reaches
    # the checkout.
    df["exit"] = df["location"] == "checkout"

    # Customer count change
    df["customer_count_change"] = 0
    df.loc[df["entry"], "customer_count_change"] = 1
    df.loc[df["exit"], "customer_count_change"] = -1
    return df


def filter_non_exiting_customers(df: pd.DataFrame) -> pd.DataFrame:
    """Filter out customers that don't exit the store before it closes.

    Some customers don't reach the checkout before the store closes and
    may be considered to involve observation.

    Parameters
    ----------
    df
        Dataframe containing at least the ``timestamp``,
        ``customer_no`` and ``exit`` columns.

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe with customers that don't exit the store filtered out.
    """
    df.sort_values(by="timestamp", inplace=True)
    df["last_appearance"] = ~df.iloc[::-1]["customer_no"].duplicated()

    invalid_customers = df.loc[df["last_appearance"] & ~df["exit"], "customer_no"]

    df.drop(columns="last_appearance", inplace=True)
    return df[~df["customer_no"].isin(invalid_customers)].reset_index(drop=True)


def customers_by_location(
    df: pd.DataFrame,
    locations: Union[str, Iterable[str]] = (
        "checkout",
        "dairy",
        "drinks",
        "fruit",
        "spices",
    ),
) -> pd.DataFrame:
    """Return dataframe containing the number of customers by location over
    time.

    Parameters
    ----------
    df
        Dataframe containing at least the ``timestamp`` and ``locations``
        columns.

    locations
        List of locations

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing the number of customers by location over
        time. E.g::

            timestamp            spices  dairy
            -------------------  ------  -----
            2019-09-02 07:03:00       0      2
            2019-09-02 07:04:00       3      2
            2019-09-02 07:05:00       0      1
    """
    locations = (
        list(locations) if isinstance(locations, collections.Iterable) else [locations]
    )

    customers_by_location = (
        df.groupby(["timestamp", "location"]).size().unstack(fill_value=0)
    )
    return customers_by_location.loc[:, locations]


def compute_customer_time_in_store(df: pd.DataFrame) -> pd.Series:
    """Return series containing the customers' total time in the store.

    Parameters
    ----------
    df
        Dataframe containing at least the ``timestamp``,
        ``customer_no``, ``entry`` and ``exit`` columns.

    Notes
    -----
    The ``entry`` and ``exit`` columns can be obtained through
    :func:`add_entry_exit`.

    Returns
    -------
    :class:`pandas.Series`
         Series containing the total time each customer spent in the
         store. E.g::

            customer_no
            -----------  --------
            1            00:02:00
            2            00:03:00
    """
    return df.groupby("customer_no").apply(
        lambda customer: customer[customer["exit"]].iloc[0].timestamp
        - customer[customer["entry"]].iloc[0].timestamp
    )


def compute_customer_total(df: pd.DataFrame) -> pd.DataFrame:
    """Return dataframe containing the total number of customers.

    Parameters
    ----------
    df
        Dataframe containing at least the ``timestamp`` and
        ``customer_count_change`` columns.

    Notes
    -----
    The ``customer_count_change`` column can be obtained through
    :func:`add_entry_exit`

    Returns
    -------
    :class:`pandas.DataFrame`
        Dataframe containing the total number of customers present in
        the store at the end of each time period. E.g::

            timestamp            customer_total
            -------------------  --------------
            2019-09-02 07:03:00               2
            2019-09-02 07:04:00               8
            2019-09-02 07:05:00               6
    """
    df["customer_total"] = df["customer_count_change"].cumsum()
    return df[["timestamp", "customer_total"]].groupby("timestamp").last()
