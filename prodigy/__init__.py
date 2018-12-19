# coding: utf8
from __future__ import unicode_literals

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="DeprecationWarning")

from . import recipes  # noqa
from .about import __version__  # noqa
from .components.loaders import get_stream, get_loader  # noqa
from .components.validate import get_schema  # noqa
from .core import recipe, recipe_args, get_recipe, set_recipe  # noqa
from .util import get_config, set_hashes, log   # noqa
from .app import server


def serve(recipe_name, *args, **config):
    loaded_recipe = get_recipe(recipe_name)
    if loaded_recipe:
        controller = loaded_recipe(*args)
        controller.config.update(config)
        server(controller, controller.config)
    else:
        raise ValueError("Can't find recipe {}.".format(recipe_name))
