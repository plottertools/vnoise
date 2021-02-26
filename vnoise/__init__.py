from .vnoise import Noise
from .vsnoise import SNoise


def _get_version() -> str:
    import pkg_resources

    return pkg_resources.get_distribution("vnoise").version


__version__ = _get_version()
