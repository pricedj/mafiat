from pkg_resources import DistributionNotFound
from pkg_resources import get_distribution
try:
    __version__ = get_distribution(__name__).version
except DistributionNotFound:
    pass  # package is not installed
