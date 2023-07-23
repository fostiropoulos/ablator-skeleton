import argparse
from pathlib import Path
import time
from omegaconf import OmegaConf
import torch
from ablator import (
    Literal,
    ModelConfig,
    ModelWrapper,
    RunConfig,
    ParallelTrainer,
    configclass,
    ParallelConfig,
)
from torch import nn
from ablator_skeleton import PACKAGE_DIR

N_BATCHES = 10_000


class CustomModelConfig(ModelConfig):
    mock_param: int = 0
    option: Literal["option_1", "option_2"] = "option_1"


@configclass
class MyRunConfig(RunConfig):
    model_config: CustomModelConfig


@configclass
class MyParallelConfig(ParallelConfig):
    model_config: CustomModelConfig


class TestWrapper(ModelWrapper):
    def make_dataloader_train(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(N_BATCHES)]
        return dl

    def make_dataloader_val(self, run_config: RunConfig):
        dl = [torch.rand(100) for i in range(N_BATCHES)]
        return dl


class MyCustomModel(nn.Module):
    def __init__(self, config: CustomModelConfig) -> None:
        super().__init__()
        self.param = nn.Parameter(torch.ones(100, 1))

    def forward(self, x: torch.Tensor):
        x = self.param + torch.rand_like(self.param) * 0.01
        return {"preds": x}, x.sum().abs()


def read_configs(path, mp_path):
    config = OmegaConf.create(Path(path).read_text(encoding="utf-8"))

    c = OmegaConf.create(Path(mp_path).read_text(encoding="utf-8"))
    config.merge_with(c)
    return OmegaConf.to_object(config)


def make_args():
    args = argparse.ArgumentParser()
    base_configs = [c.stem for c in PACKAGE_DIR.joinpath("configs").glob("*.yaml")]
    mp_configs = [c.stem for c in PACKAGE_DIR.joinpath("configs", "mp").glob("*.yaml")]
    args.add_argument(
        "--config",
        choices=base_configs,
        default="default",
    )
    args.add_argument("--mp", choices=mp_configs, default=None, required=False)

    some_options = CustomModelConfig.__annotations__["option"].__args__
    args.add_argument(
        "--option",
        choices=list(some_options),
        default=some_options[0],
        required=False,
    )
    args.add_argument(
        "--device",
        choices=["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())],
        default="cuda",
        required=False,
    )

    kwargs, _ = args.parse_known_args()
    kwargs = vars(kwargs)
    return kwargs


def run_local(
    config: str,
    option: str,
    device: str,
    **_,
):
    run_config = MyRunConfig.load(PACKAGE_DIR.joinpath("configs", f"{config}.yaml"))
    run_config.verbose = "progress"
    run_config.device = device
    run_config.model_config.random_options = option
    run_config.experiment_dir += f"/{option}_{time.time()}"
    wrapper = TestWrapper(
        model_class=MyCustomModel,
    )
    wrapper.train(run_config=run_config)


def run_mp(
    config: str,
    mp: str,
    option: str,
    **_,
):
    path = PACKAGE_DIR.joinpath("configs", f"{config}.yaml")
    mp_path = PACKAGE_DIR.joinpath("configs", "mp", f"{mp}.yaml")
    kwargs = read_configs(path, mp_path)
    run_config = MyParallelConfig(**kwargs)
    run_config.experiment_dir += f"/{option}_{time.time()}"
    run_config.device = "cuda"
    run_config.verbose = "console"
    wrapper = TestWrapper(
        model_class=MyCustomModel,
    )
    ablator = ParallelTrainer(
        wrapper=wrapper,
        run_config=run_config,
    )

    ablator.launch(working_directory=PACKAGE_DIR)


if __name__ == "__main__":
    kwargs = make_args()
    if kwargs["mp"] is not None:
        run_mp(**kwargs)
    else:
        run_local(**kwargs)
