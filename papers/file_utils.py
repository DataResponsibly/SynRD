from pathlib import Path


class PathSearcher:
    """A class for searching for paths since scripts are run from different
    locations with expected structure."""
    def __init__(self, prefix=""):
        self.prefix = prefix

    def get_path(self, path):
        candidate_paths = [
            Path(path),
            Path(self.prefix, path),
            Path("papers", self.prefix, path)
        ]

        for candidate in candidate_paths:
            if candidate.exists():
                return str(candidate.resolve())
        
        raise FileNotFoundError(f"""File doesn't exist in any of the following:
        {Path(path)}, {Path(self.prefix, path)}, {Path("papers", self.prefix, path)}.""")