# from .papers.publication import Publication
import contextlib
import importlib
import importlib.util
import inspect
import os
from pathlib import Path

from tqdm.auto import tqdm

from ..publication import Publication


def is_subclass(o):
    return inspect.isclass(o) and issubclass(o, Publication) and o != Publication


def get_papers() -> list[Publication]:
    papers = []

    cur_dir_path = str(Path(__file__).parent)
    filenames = [
        f'.{dir_name.name}.{file_name.name.removesuffix(".py")}'
        for dir_name in os.scandir(cur_dir_path)
        if dir_name.is_dir()
        for file_name in os.scandir(dir_name.path)
        if file_name.name.endswith(".py")
    ]
    for name in tqdm(filenames):
        with contextlib.suppress(Exception):
            module = importlib.import_module(name, "SynRD.papers")
            papers.extend(
                map(lambda t: t[1], inspect.getmembers(module, predicate=is_subclass))
            )

    return papers
