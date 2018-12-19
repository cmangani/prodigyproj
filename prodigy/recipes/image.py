# coding: utf8
from __future__ import unicode_literals, print_function

import copy

from ..components.loaders import get_stream
from ..components.preprocess import fetch_images
from ..core import recipe, recipe_args
from ..util import set_hashes, prints, log, b64_uri_to_bytes


@recipe('image.manual',
    dataset=recipe_args['dataset'],
    source=recipe_args['source'],
    api=recipe_args['api'],
    loader=recipe_args['loader'],
    label=recipe_args['label_set'],
    exclude=recipe_args['exclude'],
    darken=("Darken image to make boxes stand out more", "flag", "D", bool))
def image_manual(dataset, source=None, api=None, loader='images', label=None,
                 exclude=None, darken=False):
    """
    Manually annotate images by drawing rectangular bounding boxes or polygon
    shapes on the image.
    """
    log("RECIPE: Starting recipe image.manual", locals())
    stream = get_stream(source, api=api, loader=loader, input_key='image')
    stream = fetch_images(stream)

    return {
        'view_id': 'image_manual',
        'dataset': dataset,
        'stream': stream,
        'exclude': exclude,
        'config': {'labels': label, 'darken_image': 0.3 if darken else 0}
    }


@recipe('image.test',
        dataset=recipe_args['dataset'],
        lightnet_model=("Loadable lightnet model", "positional", None, str),
        source=recipe_args['source'],
        api=recipe_args['api'],
        exclude=recipe_args['exclude'])
def image_test(dataset, lightnet_model, source=None, api=None, exclude=None):
    """
    Test Prodigy's image annotation interface with a YOLOv2 model loaded
    via LightNet. Requires the LightNet library to be installed. The recipe
    will find objects in the images, and create a task for each object.
    """
    log("RECIPE: Starting recipe image.test", locals())
    try:
        import lightnet
    except ImportError:
        prints("Can't find LightNet", "In order to use this recipe, you "
               "need to have LightNet installed (currently compatible with "
               "Mac and Linux): pip install lightnet. For more details, see: "
               "https://github.com/explosion/lightnet", error=True, exits=1)

    def get_image_stream(model, stream, thresh=0.5):
        for eg in stream:
            if not eg['image'].startswith('data'):
                msg = "Expected base64-encoded data URI, but got: '{}'."
                raise ValueError(msg.format(eg['image'][:100]))
            image = lightnet.Image.from_bytes(b64_uri_to_bytes(eg['image']))
            boxes = [b for b in model(image, thresh=thresh) if b[2] >= thresh]
            eg['width'] = image.width
            eg['height'] = image.height
            eg['spans'] = [get_span(box) for box in boxes]
            for i in range(len(eg['spans'])):
                task = copy.deepcopy(eg)
                task['spans'][i]['hidden'] = False
                task = set_hashes(task, overwrite=True)
                score = task['spans'][i]['score']
                task['score'] = score
                yield task

    def get_span(box, hidden=True):
        class_id, name, prob, abs_points = box
        name = str(name, 'utf8') if not isinstance(name, str) else name
        x, y, w, h = abs_points
        rel_points = [[x - w/2, y - h/2], [x - w/2, y + h/2],
                      [x + w/2, y + h/2], [x + w/2, y - h/2]]
        return {'score': prob, 'label': name, 'label_id': class_id,
                'points': rel_points, 'center': [abs_points[0], abs_points[1]],
                'hidden': hidden}

    model = lightnet.load(lightnet_model)
    log("RECIPE: Loaded LightNet model {}".format(lightnet_model))
    stream = get_stream(source, api=api, loader='images', input_key='image')
    stream = fetch_images(stream)

    def free_lighnet(ctrl):
        nonlocal model
        del model

    return {
        'view_id': 'image',
        'dataset': dataset,
        'stream': get_image_stream(model, stream),
        'exclude': exclude,
        'on_exit': free_lighnet
    }
