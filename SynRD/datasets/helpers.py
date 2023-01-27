from abc import ABCMeta as NativeABCMeta
from typing import Any, Callable, TypeVar, cast

from tqdm.auto import tqdm

R = TypeVar("R")


class DummyAttribute:
    pass


def abstract_attribute(obj: Callable[[Any], R] | None = None) -> R:
    _obj = cast(Any, obj)
    if obj is None:
        _obj = DummyAttribute()
    _obj.__is_abstract_attribute__ = True
    return cast(R, _obj)


class ABCMeta(NativeABCMeta):
    def __call__(self, *args, **kwargs):
        instance = NativeABCMeta.__call__(self, *args, **kwargs)
        if abstract_attributes := {
            name
            for name in dir(instance)
            if getattr(getattr(instance, name), "__is_abstract_attribute__", False)
        }:
            raise NotImplementedError(
                f"Can't instantiate abstract class {self.__name__} with abstract attributes: {', '.join(abstract_attributes)}"
            )
        return instance


class URLRetrieveTQDM:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = tqdm(total=total_size)

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.n = downloaded
            self.pbar.refresh()
        else:
            self.pbar.close()
            self.pbar = None
