from pathlib import Path


def get_project_root() -> Path:
    """

    :rtype: object
    """
    return Path(__file__).parent.parent
