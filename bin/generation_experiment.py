import logging
import argparse
import math
from pathlib import Path

import yaml
import numpy as np
import torch
from tqdm.auto import tqdm

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
from uncond_ts_diff.model import TSDiff
import uncond_ts_diff.configs as diffusion_configs


def load_model(config):
    model = TSDiff(
        **getattr(
            diffusion_configs,
            config.get("diffusion_config", "diffusion_small_config"),
        ),
        freq=config["freq"],
        use_features=config["use_features"],
        use_lags=config["use_lags"],
        normalization="mean",
        context_length=config["context_length"],
        prediction_length=config["prediction_length"],
        init_skip=config["init_skip"],
    )
    model.load_state_dict(
        torch.load(config["ckpt"], map_location="cpu"),
        strict=True,
    )
    model = model.to(config["device"])
    return model


def sample_synthetic(
    model: TSDiff,
    num_samples: int = 10_000,
    batch_size: int = 1000,
):
    synth_samples = []

    n_iters = math.ceil(num_samples / batch_size)
    for _ in tqdm(range(n_iters)):
        samples = model.sample_n(num_samples=batch_size)
        synth_samples.append(samples)

    synth_samples = np.concatenate(synth_samples, axis=0)[:num_samples]

    return synth_samples


def sample_real(
    data_loader,
    n_timesteps: int,
    num_samples: int = 10_000,
    batch_size: int = 1000,
):
    real_samples = []
    data_iter = iter(data_loader)
    n_iters = math.ceil(num_samples / batch_size)
    for _ in tqdm(range(n_iters)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            batch = next(data_iter)
        ts = np.concatenate(
            [batch["past_target"], batch["future_target"]], axis=-1
        )[:, -n_timesteps:]
        real_samples.append(ts)

    real_samples = np.concatenate(real_samples, axis=0)[:num_samples]

    return real_samples

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
        num_samples=1000,
    )
    np.save(log_dir / "real_samples.npy", real_samples)

    if samples_path is None:
        # Generate synthetic samples
        logger.info("Generating synthetic samples")
        synth_samples = sample_synthetic(model, num_samples=1000)
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
