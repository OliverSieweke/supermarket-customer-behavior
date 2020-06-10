"""
Paths
=====

This module provides utility methods for retrieving project paths.
"""

# Standard Library ---------------------------------------------------------------------
from enum import Enum
from pathlib import Path


def project_root_path() -> Path:
    """Return absolute project root path.

    Returns
    -------
    :class:`pathlib.Path`
        Absolute project root path.
    """
    return Path(__file__).resolve().parents[2]


def root_module_path() -> Path:
    """Return absolute root module path.

    Returns
    -------
    :class:`pathlib.Path`
        Absolute root module path.
    """
    return Path(__file__).resolve().parents[1]


def data_dir_path() -> Path:
    """Return absolute data directory path.

    Returns
    -------
    :class:`pathlib.Path`
        Absolute data directory path.
    """
    return project_root_path().joinpath("data")


class WeekDay(Enum):
    """Enum of week days for which data is available."""

    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"


def day_data_file_path(day: WeekDay) -> Path:
    """Return absolute day data file path.

    Returns
    -------
    :class:`pathlib.Path`
        Day data file path.
    """
    return data_dir_path().joinpath(f"{day.value}.csv")
