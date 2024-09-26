from pathlib import Path

from gluonts.dataset.common import (
    MetaData,
    TrainDatasets,
    FileDataset,
)


def get_custom_dataset(
    jsonl_path: Path,
    freq: str,
    prediction_length: int,
):
    """Creates a custom GluonTS dataset from a JSONLines file and
    give parameters.

    Parameters
    ----------
    jsonl_path
        Path to a JSONLines file with time series
    freq
        Frequency in pandas format
        (e.g., `H` for hourly, `D` for daily)
    prediction_length
        Prediction length

    Returns
    -------
        A gluonts dataset
    """
    metadata = MetaData(freq=freq, prediction_length=prediction_length)
    train_ts = FileDataset(jsonl_path, freq)
    dataset = TrainDatasets(metadata=metadata, train=train_ts)
    return dataset
