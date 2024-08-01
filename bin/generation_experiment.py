import logging
import argparse
from pathlib import Path

import yaml
import numpy as np

from gluonts.dataset.loader import TrainDataLoader
from gluonts.itertools import Cached
from gluonts.torch.batchify import batchify
from gluonts.time_feature import (
    time_features_from_frequency_str,
)

from uncond_ts_diff.utils import (
    create_transforms,
    create_splitter,
    get_next_file_num,
    add_config_to_argparser,
)
from uncond_ts_diff.custom_dataset import get_custom_dataset

from tstr_experiment import load_model, sample_synthetic, sample_real


def main(config: dict, log_dir: str, samples_path: str):
    # Read global parameters
    dataset_name = config["dataset"]
    context_length = config["context_length"]
    prediction_length = config["prediction_length"]

    # Create log_dir
    log_dir: Path = Path(log_dir)
    base_dirname = "generation_log"
    run_num = get_next_file_num(
        base_dirname, log_dir, file_type="", separator="-"
    )
    log_dir = log_dir / f"{base_dirname}-{run_num}"
    log_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"Logging to {log_dir}")

    # Load dataset and model
    logger.info("Loading model")
    dataset = get_custom_dataset(config['dataset_path'], 'D', prediction_length)
    config["freq"] = dataset.metadata.freq
    assert prediction_length == dataset.metadata.prediction_length

    model = load_model(config)

    # Setup data transformation and loading
    transformation = create_transforms(
        num_feat_dynamic_real=0,
        num_feat_static_cat=0,
        num_feat_static_real=0,
        time_features=time_features_from_frequency_str(config["freq"]),
        prediction_length=prediction_length,
    )
    transformed_data = transformation.apply(list(dataset.train), is_train=True)
    training_splitter = create_splitter(
        past_length=context_length + max(model.lags_seq),
        future_length=prediction_length,
        mode="train",
    )
    train_dataloader = TrainDataLoader(
        Cached(transformed_data),
        batch_size=1000,
        stack_fn=batchify,
        transform=training_splitter,
    )

    # Generate real samples
    logger.info("Generating real samples")
    real_samples = sample_real(
        train_dataloader,
        n_timesteps=context_length + prediction_length,
        num_samples=10000,
    )
    np.save(log_dir / "real_samples.npy", real_samples)

    if samples_path is None:
        # Generate synthetic samples
        logger.info("Generating synthetic samples")
        synth_samples = sample_synthetic(model, num_samples=10000)
        np.save(log_dir / "synth_samples.npy", synth_samples)
    else:
        logger.info(f"Using synthetic samples from {samples_path}")
        synth_samples = np.load(samples_path)[:10000]
        synth_samples = synth_samples.reshape(
            (10000, context_length + prediction_length)
        )



if __name__ == "__main__":
    # Setup Logger
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.INFO)

    # Setup argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to yaml config"
    )
    parser.add_argument(
        "--out_dir", type=str, default="./results", help="Path to results dir"
    )
    parser.add_argument(
        "--samples_path", type=str, help="Path to generated samples"
    )
    args, _ = parser.parse_known_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    # Update config from command line
    parser = add_config_to_argparser(config=config, parser=parser)
    args = parser.parse_args()
    config_updates = vars(args)
    for k in config.keys() & config_updates.keys():
        orig_val = config[k]
        updated_val = config_updates[k]
        if updated_val != orig_val:
            logger.info(f"Updated key '{k}': {orig_val} -> {updated_val}")
    config.update(config_updates)

    main(config=config, log_dir=args.out_dir, samples_path=args.samples_path)
