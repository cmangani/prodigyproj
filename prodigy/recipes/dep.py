# coding: utf8
from __future__ import unicode_literals

import random
import spacy

from ..models.dep import DependencyParser, merge_arcs
from ..components import printers
from ..components.db import connect
from ..components.preprocess import split_sentences, add_tokens
from ..components.sorters import prefer_uncertain
from ..components.loaders import get_stream
from ..core import recipe, recipe_args
from ..util import split_evals, get_print, export_model_data, log


@recipe('dep.teach',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        label=recipe_args['label_set'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        exclude=recipe_args['exclude'],
        unsegmented=recipe_args['unsegmented'])
def teach(dataset, spacy_model, label=None, source=None, api=None, loader=None,
          exclude=None, unsegmented=False):
    """
    Collect the best possible training data for a dependency parsing model with
    the model in the loop. Based on your annotations, Prodigy will decide which
    questions to ask next.
    """
    log("RECIPE: Starting recipe dep.teach", locals())
    # Initialize the stream, and ensure that hashes are correct, and examples
    # are deduplicated.
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    # Create the model, using a pre-trained spaCy model.
    nlp = spacy.load(spacy_model, disable=['ner', 'tagger'])
    log("RECIPE: Creating DependencyParser using model {}".format(spacy_model))
    model = DependencyParser(nlp, label=label)
    # Split the stream into sentences
    if not unsegmented:
        stream = split_sentences(model.orig_nlp, stream)
    # Add "tokens" to stream
    stream = add_tokens(model.orig_nlp, stream)
    # Return components, to construct Controller
    return {
        'view_id': 'dep',
        'dataset': dataset,
        'stream': prefer_uncertain(model(stream)),
        'update': model.update,  # callback to update the model in-place
        'exclude': exclude,
        'config': {'lang': model.nlp.lang}
    }


@recipe('dep.batch-train',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        label=recipe_args['label_set'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        beam_width=recipe_args['beam_width'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        silent=recipe_args['silent'])
def batch_train(dataset, input_model, output_model=None, label='', factor=1,
                dropout=0.2, n_iter=10, batch_size=32, beam_width=16,
                eval_id=None, eval_split=None, silent=False):
    """
    Batch train a dependency parsing model from annotations. Prodigy will
    export the best result to the output directory, and include a JSONL file of
    the training and evaluation examples. You can either supply a dataset ID
    containing the evaluation data, or choose to split off a percentage of
    examples for evaluation.
    """
    log("RECIPE: Starting recipe dep.batch-train", locals())
    DB = connect()
    if not dataset in DB:
        prints("Can't find dataset '{}'".format(dataset), exits=1, error=True)
    print_ = get_print(silent)
    random.seed(0)
    nlp = spacy.load(input_model)
    print_("\nLoaded model {}".format(input_model))
    examples = merge_arcs(DB.get_dataset(dataset))
    random.shuffle(examples)
    if 'parser' not in nlp.pipe_names:
        parser = nlp.create_pipe('parser')
        for eg in examples:
            for arc in eg.get('arcs', []):
                parser.add_label(arc['label'])
        nlp.add_pipe(parser, last=True)
        nlp.begin_training()
    else:
        parser = nlp.get_pipe('parser')
        for l in label:
            parser.add_label(l)
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of accept/reject examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    model = DependencyParser(nlp, label=label)
    other_pipes = [p for p in nlp.pipe_names if p not in ('parser', 'sbd')]
    if other_pipes:
        disabled = nlp.disable_pipes(*other_pipes)
        log("RECIPE: Temporarily disabled other pipes: {}".format(other_pipes))
    else:
        disabled = None
    log('RECIPE: Initialised DependencyParser with model {}'
        .format(input_model), model.nlp.meta)
    baseline = model.evaluate(evals)
    log("RECIPE: Calculated baseline from evaluation examples "
        "(accuracy %.2f)" % baseline['acc'])
    best = None
    random.shuffle(examples)
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    print_(printers.dep_before(**baseline))
    if len(evals) > 0:
        print_(printers.dep_update_header())

    for i in range(n_iter):
        losses = model.batch_train(examples, batch_size=batch_size,
                                   drop=dropout, beam_width=beam_width)
        stats = model.evaluate(evals)
        if best is None or stats['acc'] > best[0]:
            model_to_bytes = None
            if output_model is not None:
                model_to_bytes = model.to_bytes()
            best = (stats['acc'], stats, model_to_bytes)
        print_(printers.dep_update(i, losses, stats))
    best_acc, best_stats, best_model = best
    print_(printers.dep_result(best_stats, best_acc, baseline['acc']))
    if output_model is not None:
        model.from_bytes(best_model)
        if disabled:
            log("RECIPE: Restoring disabled pipes: {}".format(other_pipes))
            disabled.restore()
        msg = export_model_data(output_model, model.nlp, examples, evals)
        print_(msg)
    best_stats['baseline'] = baseline['acc']
    best_stats['acc'] = best_acc
    return best_stats


@recipe('dep.train-curve',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        label=recipe_args['label_set'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        beam_width=recipe_args['beam_width'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        n_samples=recipe_args['n_samples'])
def train_curve(dataset, input_model, label='', dropout=0.2, n_iter=5,
                batch_size=32, beam_width=16, eval_id=None, eval_split=None,
                n_samples=4):  # pragma: no cover
    """
    Batch-train models with different portions of the training examples and
    print the accuracy figures and accuracy improvements.
    """
    log("RECIPE: Starting recipe dep.train-curve", locals())
    factors = [(i + 1) / n_samples for i in range(n_samples)]
    prev_acc = 0
    print("\nStarting with model {}".format(input_model))
    print(printers.trainconf(dropout, n_iter, batch_size, samples=n_samples))
    print(printers.dep_curve_header())
    for factor in factors:
        best_stats = batch_train(dataset, input_model=input_model, label=label,
                                 factor=factor, dropout=dropout,
                                 n_iter=n_iter, batch_size=batch_size,
                                 beam_width=beam_width, eval_id=eval_id,
                                 eval_split=eval_split, silent=True)
        print(printers.dep_curve(factor, best_stats, prev_acc))
        prev_acc = best_stats['acc']
