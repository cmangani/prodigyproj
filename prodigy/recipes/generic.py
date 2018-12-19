# coding: utf8
from __future__ import unicode_literals, print_function

from collections import Counter

from ..components import printers
from ..components.loaders import get_stream
from ..core import recipe, recipe_args
from ..util import TASK_HASH_ATTR, log


@recipe('mark',
        dataset=recipe_args['dataset'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label'],
        view_id=recipe_args['view'],
        memorize=recipe_args['memorize'],
        exclude=recipe_args['exclude'])
def mark(dataset, source=None, view_id=None, label='', api=None,
         loader=None, memorize=False, exclude=None):
    """
    Click through pre-prepared examples, with no model in the loop.
    """
    log('RECIPE: Starting recipe mark', locals())
    stream = get_stream(source, api, loader)
    counts = Counter()
    memory = {}

    def fill_memory(ctrl):
        if memorize:
            examples = ctrl.db.get_dataset(dataset)
            log("RECIPE: Add {} examples from dataset '{}' to memory"
                .format(len(examples), dataset))
            for eg in examples:
                memory[eg[TASK_HASH_ATTR]] = eg['answer']

    def ask_questions(stream):
        for eg in stream:
            if TASK_HASH_ATTR in eg and eg[TASK_HASH_ATTR] in memory:
                answer = memory[eg[TASK_HASH_ATTR]]
                counts[answer] += 1
            else:
                if label:
                    eg['label'] = label
                yield eg

    def recv_answers(answers):
        for eg in answers:
            counts[eg['answer']] += 1
            memory[eg[TASK_HASH_ATTR]] = eg['answer']

    def print_results(ctrl):
        print(printers.answers(counts))

    return {
        'view_id': view_id,
        'dataset': dataset,
        'stream': ask_questions(stream),
        'exclude': exclude,
        'update': recv_answers,
        'on_load': fill_memory,
        'on_exit': print_results,
        'config': {'label': label}
    }
