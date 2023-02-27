from typing import List, Iterator

from classy.data.data_drivers import ClassySample, get_data_driver, DataDriver
from classy.utils.log import get_project_logger

from src.data.data_drivers.base import ExmodDataDriver, ExmodSample

logger = get_project_logger(__name__)


@DataDriver.register("generation", "mix")
class AlternatorDataDriver(ExmodDataDriver):
    def _read_config_from_path(self, path: str):
        config = fix_paths(
            OmegaConf.load(path),
            check_fn=lambda path: os.path.exists(
                hydra.utils.to_absolute_path(path[: path.rindex("/")])
            ),
            fix_fn=lambda path: hydra.utils.to_absolute_path(path),
        )
        return config

    def get_unique_sentences(self, path: str) -> List[str]:
        config = self._read_config_from_path(path)
        for _config in config:
            _data_driver = get_data_driver(**_config.data_driver_specs)
            yield from _data_driver.get_unique_sentences(_config.path)

    def dataset_exists_at_path(self, path: str) -> bool:
        config = self._read_config_from_path(path)
        return all(
            get_data_driver(**_config.data_driver_specs).dataset_exists_at_path(
                _config.path
            )
            for _config in config
        )

    def read_from_path(self, path: str) -> Iterator[ExmodSample]:

        config = self._read_config_from_path(path)
        dds = [get_data_driver(**_config.data_driver_specs) for _config in config]
        paths = [_config.path for _config in config]

        ps = [_config.p for _config in config]
        ps = [p / sum(ps) for p in ps]

        its = [iter(dds[i].read_from_path(paths[i])) for i in range(len(config))]
        done = [False for i in range(len(config))]

        while True:

            if all(_done for _done in done):
                break

            i = np.random.choice(len(its), p=ps)

            try:
                sample = next(its[i])
            except StopIteration:
                logger.info(f"Data driver {i} completed. Resetting it.")
                done[i] = True
                its[i] = iter(dds[i].read_from_path(paths[i]))

            yield sample

    def save(
        self,
        samples: Iterator[ClassySample],
        path: str,
        use_predicted_annotation: bool = False,
    ):
        raise NotImplementedError


if __name__ == "__main__":
    from src.data.data_drivers.raganato.data_driver import *
    from src.data.data_drivers.base import *

    dd = get_data_driver("generation", "mix")

    print(
        len(
            list(
                dd.get_unique_sentences(
                    "data/training-configurations/exmod/k1+fews/train.mix"
                )
            )
        )
    )

    for sample in dd.read_from_path(
        "data/training-configurations/exmod/k1+fews/train.mix"
    ):
        print(sample)
        break
