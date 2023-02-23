from .saw2018cross import Saw2018Cross
from .fairman2019marijuana import Fairman2019Marijuana
from .iverson22football import Iverson22Football
from .jeong2021math import Jeong2021Math
from .fruiht2018naturally import Fruiht2018Naturally
from .lee2021ability import Lee2021Ability
from .pierce2019who  import Pierce2019Who
__all__ = ["Saw2018Cross", 
            "Iverson22Football",
            "Fairman2019Marijuana",
            "Jeong2021Math",
            "Fruiht2018Naturally",
            "Lee2021Ability",
            "Pierce2019Who"]

# import contextlib
# import importlib
# import importlib.util
# import inspect
# import os
# from pathlib import Path

# from tqdm.auto import tqdm

# from ..publication import Publication


# def is_subclass(o):
#     return inspect.isclass(o) and issubclass(o, Publication) and o != Publication


# def get_papers() -> list[Publication]:
#     papers = []

#     cur_dir_path = str(Path(__file__).parent)
#     filenames = [
#         f'.{dir_name.name}.{file_name.name.removesuffix(".py")}'
#         for dir_name in os.scandir(cur_dir_path)
#         if dir_name.is_dir()
#         for file_name in os.scandir(dir_name.path)
#         if file_name.name.endswith(".py")
#     ]
#     for name in tqdm(filenames):
#         with contextlib.suppress(Exception):
#             module = importlib.import_module(name, "SynRD.papers")
#             papers.extend(
#                 map(lambda t: t[1], inspect.getmembers(module, predicate=is_subclass))
#             )

#     return papers