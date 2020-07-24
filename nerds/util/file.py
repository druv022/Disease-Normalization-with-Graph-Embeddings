import shutil
from pathlib import Path


def mkdir(directory, parents=True):
    """ Makes a directory after checking whether it already exists.

        Parameters:
            directory (str | Path): The name of the directory to be created.
            parents (boolean): If True, then parent directories are created as well
    """
    path_dir = Path(directory)
    if not path_dir.exists():
        path_dir.mkdir(parents=parents)


def rmdir(directory, recursive=False):
    """ Removes an empty directory after checking whether it already exists.

        Parameters:
            directory (str | Path): The name of the directory to be removed.
            recursive (boolean): If True, then the contents are removed (including subdirectories and files),
                otherwise, the directory is removed only if it is empty
    """
    path_dir = Path(directory)
    if not path_dir.exists():
        return
    if recursive:
        shutil.rmtree(path_dir)
    else:
        if len(list(path_dir.iterdir())) == 0:
            path_dir.rmdir()
        else:
            raise ValueError(
                "Cannot remove directory '{}' as it is not empty: consider removing it recursively".format(path_dir))
