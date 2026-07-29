"""Tabular and time-series datasets.

Distinguished from text corpora, which use the same file formats. A folder of Parquet is
a language corpus or a table of sensor readings depending on what is in it, and since the
inventory cannot read the columns, it goes by the vocabulary of the name — which is what a
person does too.
"""

from __future__ import annotations

from ai_asset_manager.backend.taxonomy.plugins._shared import (
    IMAGE_EXTENSIONS,
    TABULAR_EXTENSIONS,
    family_of,
    is_dataset,
)
from ai_asset_manager.backend.taxonomy.registry import TaxonomyRegistry
from ai_asset_manager.backend.taxonomy.types import (
    CONFIDENCE_STRONG,
    AssetProfile,
    Category,
    Classification,
    Task,
)

#: Words that mean a table of measurements rather than a corpus of prose.
TIMESERIES_MARKERS = ("time-series", "timeseries", "time_series", "forecast", "telemetry",
                      "sensor", "iot", "ecg", "eeg", "seismic", "weather", "traffic-flow",
                      "electricity", "stock", "ohlcv", "ticker", "candlestick")

TABULAR_MARKERS = ("tabular", "credit", "fraud", "churn", "titanic", "adult-census",
                   "kaggle", "housing", "insurance", "transactions")

#: Named benchmarks in the forecasting literature.
TIMESERIES_FAMILIES = (
    ("Monash", ("monash",)),
    ("M4", ("m4-competition", "m4_dataset")),
    ("ETT", ("etth1", "etth2", "ettm1", "ettm2")),
    ("Electricity", ("electricity-load", "ld2011")),
    ("Traffic", ("pems", "metr-la")),
    ("UCI", ("uci-",)),
)

TASKS = (
    Task(id="forecasting", label="Forecasting", domain="timeseries", order=10),
    Task(id="time_series_classification", label="Time Series Classification",
         domain="timeseries", order=20),
    Task(id="tabular_classification", label="Tabular Classification",
         domain="tabular", order=30),
    Task(id="tabular_regression", label="Tabular Regression", domain="tabular", order=40),
)


def register(registry: TaxonomyRegistry) -> None:
    """Register the tabular shelf and its classifier."""
    for task in TASKS:
        registry.add_task(task)

    registry.add_category(
        Category(id="tabular_dataset", label="Tabular / Time Series", section="datasets",
                 order=275, domain="tabular",
                 aliases=("tabular-datasets", "timeseries", "time-series"))
    )

    registry.add_classifier(_tabular_dataset, name="tabular.dataset", priority=400)


def _tabular_dataset(profile: AssetProfile) -> Classification | None:
    """Claim record data that is measurements rather than language."""
    if not is_dataset(profile):
        return None

    timeseries = profile.matches(TIMESERIES_MARKERS)
    tabular = profile.matches(TABULAR_MARKERS)
    family = family_of(profile, TIMESERIES_FAMILIES)

    if timeseries is None and tabular is None and family is None:
        return None

    # Images beside the records mean this is a vision dataset with a metadata table, not a
    # tabular dataset.
    if profile.files.loaded and profile.files.count(*IMAGE_EXTENSIONS) > 16:
        return None
    if profile.files.loaded and not profile.files.count(*TABULAR_EXTENSIONS):
        return None

    marker = timeseries or family
    return Classification(
        category="tabular_dataset",
        task="forecasting" if marker else "tabular_classification",
        domain="timeseries" if marker else "tabular",
        family=family, modalities=("tabular",), confidence=CONFIDENCE_STRONG,
        evidence=f"{family} benchmark" if family else f"name contains {timeseries or tabular!r}",
    )
