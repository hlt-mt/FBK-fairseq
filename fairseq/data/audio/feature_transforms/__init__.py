import importlib
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set


class AudioFeatureTransform(ABC):
    @classmethod
    @abstractmethod
    def from_config_dict(cls, config: Optional[Dict] = None):
        pass

    @property
    def extra_args(self) -> Set[str]:
        """
        If the transform requires other fields of the dataset (and not only the waveform),
        list here the additional properties required. They will be fed to the __call__
        function as key-value args.
        """
        return {}


AUDIO_FEATURE_TRANSFORM_REGISTRY = {}
AUDIO_FEATURE_TRANSFORM_CLASS_NAMES = set()


def register_audio_feature_transform(name):
    def register_audio_feature_transform_cls(cls):
        if name in AUDIO_FEATURE_TRANSFORM_REGISTRY:
            raise ValueError(f"Cannot register duplicate transform ({name})")
        if not issubclass(cls, AudioFeatureTransform):
            raise ValueError(
                f"Transform ({name}: {cls.__name__}) must extend "
                "AudioFeatureTransform"
            )
        if cls.__name__ in AUDIO_FEATURE_TRANSFORM_CLASS_NAMES:
            raise ValueError(
                f"Cannot register audio feature transform with duplicate "
                f"class name ({cls.__name__})"
            )
        AUDIO_FEATURE_TRANSFORM_REGISTRY[name] = cls
        AUDIO_FEATURE_TRANSFORM_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_audio_feature_transform_cls


def get_audio_feature_transform(name):
    return AUDIO_FEATURE_TRANSFORM_REGISTRY[name]


transforms_dir = os.path.dirname(__file__)
for file in os.listdir(transforms_dir):
    path = os.path.join(transforms_dir, file)
    if (
        not file.startswith("_")
        and not file.startswith(".")
        and (file.endswith(".py") or os.path.isdir(path))
    ):
        name = file[: file.find(".py")] if file.endswith(".py") else file
        importlib.import_module("fairseq.data.audio.feature_transforms." + name)


class CompositeAudioFeatureTransform(AudioFeatureTransform):
    @classmethod
    def from_config_dict(cls, config=None):
        _config = {} if config is None else config
        _transforms = _config.get("transforms")
        if _transforms is None:
            return None
        transforms = [
            get_audio_feature_transform(_t).from_config_dict(_config.get(_t))
            for _t in _transforms
        ]
        return CompositeAudioFeatureTransform(transforms)

    def __init__(self, transforms):
        self.transforms = [t for t in transforms if t is not None]
        self._extra_args_needed = set()
        for t in self.transforms:
            self._extra_args_needed = self._extra_args_needed.union(t.extra_args)

    @property
    def extra_args(self) -> Set[str]:
        return self._extra_args_needed

    def __call__(self, x, **kwargs):
        for t in self.transforms:
            t_kwargs = {k: kwargs[k] for k in t.extra_args}
            x = t(x, **t_kwargs)
        return x

    def __repr__(self):
        format_string = (
            [self.__class__.__name__ + "("]
            + [f"    {t.__repr__()}" for t in self.transforms]
            + [")"]
        )
        return "\n".join(format_string)
