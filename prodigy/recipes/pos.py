# coding: utf8
from __future__ import unicode_literals

import spacy
import ujson
import copy
import random
from pathlib import Path

from ..models.pos import Tagger, merge_tags
from ..components import printers
from ..components.loaders import get_stream
from ..components.preprocess import split_sentences, add_tokens
from ..components.sorters import prefer_uncertain
from ..components.db import connect
from ..core import recipe, recipe_args
from ..util import log, prints, read_json, write_jsonl, get_print, set_hashes
from ..util import split_evals, export_model_data, INPUT_HASH_ATTR


POS_TAGS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CONJ', 'CCONJ', 'DET', 'INTJ', 'NOUN',
            'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB',
            'X', 'SPACE']


def get_tag_map(tag_map):
    log("RECIPE: Using tag map from file", tag_map)
    if not Path(tag_map).exists():
        prints('Not a valid tag map file', tag_map, error=True, exits=1)
    tag_map = read_json(tag_map)
    return tag_map


@recipe('pos.teach',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label_set'],
        tag_map=recipe_args['tag_map'],
        exclude=recipe_args['exclude'],
        unsegmented=recipe_args['unsegmented'])
def teach(dataset, spacy_model, source=None, api=None, loader=None,
          label=None, tag_map=None, exclude=None, unsegmented=False):
    """
    Collect the best possible training data for a part-of-speech tagging
    model with the model in the loop. Based on your annotations, Prodigy will
    decide which questions to ask next.
    """
    log("RECIPE: Starting recipe pos.teach", locals())
    # Initialize the stream, and ensure that hashes are correct, and examples
    # are deduplicated.
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    # Create the model, using a pre-trained spaCy model.
    log("RECIPE: Creating Tagger using model {}".format(spacy_model))
    if tag_map is not None:
        tag_map = get_tag_map(tag_map)
    model = Tagger(spacy.load(spacy_model), label=label, tag_map=tag_map)
    # Split the stream into sentences
    if not unsegmented:
        stream = split_sentences(model.nlp, stream)
    # Tokenize the stream
    stream = add_tokens(model.nlp, stream)

    return {
        'view_id': 'pos',
        'dataset': dataset,
        'stream': prefer_uncertain(model(stream), algorithm='probability'),
        'update': model.update,
        'exclude': exclude,
        'config': {'lang': model.nlp.lang,
                   'label': (', '.join(label)) if label is not None else 'all'}
    }


@recipe('pos.make-gold',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label_set'],
        exclude=recipe_args['exclude'],
        unsegmented=recipe_args['unsegmented'],
        fine_grained=("Use fine-grained part-of-speech tags, i.e. Token.tag_ "
                      "instead of Token.pos_. Warning: Can lead to unexpected "
                      "results and very long tags for some language models "
                      "that use fine-grained tags with morphological features",
                      "flag", "FG", bool))
def make_gold(dataset, spacy_model, source=None, api=None, loader=None,
              label=None, exclude=None, unsegmented=False, fine_grained=False):
    """
    Create gold data for part-of-speech tags by correcting a model's
    predictions.
    """
    log("RECIPE: Starting recipe pos.make-gold", locals())
    nlp = spacy.load(spacy_model)
    log("RECIPE: Loaded model {}".format(spacy_model))
    # Get the label set from the `label` argument, which is either a
    # comma-separated list or a path to a text file.
    labels = label
    if not labels and not fine_grained:
        print("Using universal coarse-grained POS tags: {}"
              .format(','.join(POS_TAGS)))
        labels = POS_TAGS
    log("RECIPE: Annotating with {} labels".format(len(labels)), labels)
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    # Split the stream into sentences
    if not unsegmented:
        stream = split_sentences(nlp, stream)
    # Tokenize the stream
    stream = add_tokens(nlp, stream)

    def make_tasks(nlp, stream):
        """Add a 'spans' key to each example, with predicted tags."""
        texts = ((eg['text'], eg) for eg in stream)
        for doc, eg in nlp.pipe(texts, as_tuples=True):
            task = copy.deepcopy(eg)
            spans = []
            for i, token in enumerate(doc):
                pos_tag = token.pos_ if not fine_grained else token.tag_
                if labels and pos_tag not in labels:
                    continue
                spans.append({
                    'token_start': i,
                    'token_end': i,
                    'start': token.idx,
                    'end': token.idx + len(token.text),
                    'text': token.text,
                    'label': pos_tag,
                    'source': spacy_model,
                    'input_hash': eg[INPUT_HASH_ATTR]
                })
            task['spans'] = spans
            task = set_hashes(task)
            yield task

    return {
        'view_id': 'pos_manual',
        'dataset': dataset,
        'stream': make_tasks(nlp, stream),
        'exclude': exclude,
        'update': None,
        'config': {'lang': nlp.lang, 'labels': labels}
    }


@recipe('pos.batch-train',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        output_model=recipe_args['output'],
        label=recipe_args['label_set'],
        tag_map=recipe_args['tag_map'],
        factor=recipe_args['factor'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        unsegmented=recipe_args['unsegmented'],
        silent=recipe_args['silent'])
def batch_train(dataset, input_model, output_model=None, label='', tag_map=None,
                factor=1, dropout=0.2, n_iter=10, batch_size=4, eval_id=None,
                eval_split=None, unsegmented=False,
                silent=False):
    """
    Batch train a tagging model from annotations. Prodigy will
    export the best result to the output directory, and include a JSONL file of
    the training and evaluation examples. You can either supply a dataset ID
    containing the evaluation data, or choose to split off a percentage of
    examples for evaluation.
    """
    log("RECIPE: Starting recipe pos.batch-train", locals())
    DB = connect()
    if not dataset in DB:
        prints("Can't find dataset '{}'".format(dataset), exits=1, error=True)
    print_ = get_print(silent)
    random.seed(0)
    nlp = spacy.load(input_model)
    print_("\nLoaded model {}".format(input_model))
    if 'sentencizer' not in nlp.pipe_names and 'sbd' not in nlp.pipe_names:
        nlp.add_pipe(nlp.create_pipe('sentencizer'), first=True)
        log("RECIPE: Added sentence boundary detector to model pipeline",
            nlp.pipe_names)
    examples = merge_tags(DB.get_dataset(dataset))
    random.shuffle(examples)
    if 'tagger' not in nlp.pipe_names:
        tagger = nlp.create_pipe('tagger')
    else:
        tagger = nlp.get_pipe('tagger')
    if 'tagger' not in nlp.pipe_names:
        nlp.add_pipe(tagger, last=True)
        nlp.begin_training()
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of accept/reject examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    if tag_map is not None:
        tag_map = get_tag_map(tag_map)
    model = Tagger(nlp, label=label, tag_map=tag_map)
    other_pipes = [p for p in nlp.pipe_names if p not in ('tagger', 'sbd')]
    if other_pipes:
        disabled = nlp.disable_pipes(*other_pipes)
        log("RECIPE: Temporarily disabled other pipes: {}".format(other_pipes))
    else:
        disabled = None
    log('RECIPE: Initialised Tagger with model {}'
        .format(input_model), model.nlp.meta)
    if not unsegmented:
        examples = list(split_sentences(model.nlp, examples))
        evals = list(split_sentences(model.nlp, evals))
    else:
        examples = list(examples)
        evals = list(evals)
    baseline = model.evaluate(evals)
    log("RECIPE: Calculated baseline from evaluation examples "
        "(accuracy %.2f)" % baseline['acc'])
    best = None
    random.shuffle(examples)
    examples = examples[:int(len(examples) * factor)]
    print_(printers.trainconf(dropout, n_iter, batch_size, factor,
                              len(examples)))
    print_(printers.pos_before(**baseline))
    if len(evals) > 0:
        print_(printers.pos_update_header())

    for i in range(n_iter):
        losses = model.batch_train(examples, batch_size=batch_size,
                                   drop=dropout)
        stats = model.evaluate(evals)
        if best is None or stats['acc'] > best[0]:
            model_to_bytes = None
            if output_model is not None:
                model_to_bytes = model.to_bytes()
            best = (stats['acc'], stats, model_to_bytes)
        print_(printers.pos_update(i, losses, stats))
    best_acc, best_stats, best_model = best
    print_(printers.ner_result(best_stats, best_acc, baseline['acc']))
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


@recipe('pos.train-curve',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        label=recipe_args['entity_label'],
        tag_map=recipe_args['tag_map'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        beam_width=recipe_args['beam_width'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        unsegmented=recipe_args['unsegmented'],
        n_samples=recipe_args['n_samples'])
def train_curve(dataset, input_model, label='', tag_map=None, dropout=0.2, n_iter=5,
                batch_size=32, beam_width=16, eval_id=None, eval_split=None,
                unsegmented=False, n_samples=4):
    """
    Batch-train models with different portions of the training examples and
    print the accuracy figures and accuracy improvements.
    """
    log("RECIPE: Starting recipe pos.train-curve", locals())
    factors = [(i + 1) / n_samples for i in range(n_samples)]
    prev_acc = 0
    print("\nStarting with model {}".format(input_model))
    print(printers.trainconf(dropout, n_iter, batch_size, samples=n_samples))
    print(printers.pos_curve_header())
    for factor in factors:
        best_stats = batch_train(dataset, input_model=input_model, label=label,
                                 tag_map=tag_map, factor=factor, dropout=dropout,
                                 n_iter=n_iter, batch_size=batch_size,
                                 eval_id=eval_id, eval_split=eval_split,
                                 unsegmented=unsegmented, silent=True)
        print(printers.pos_curve(factor, best_stats, prev_acc))
        prev_acc = best_stats['acc']


@recipe('pos.gold-to-spacy',
        dataset=recipe_args['dataset'],
        output_file=recipe_args['output_file'])
def gold_to_spacy(dataset, output_file=None):
    """
    Convert a dataset with annotated part-of-speech tags to the format required
    to train spaCy's part-of-speech tagger. Will export a JSONL file with one
    entry per line: ["I like eggs", {"tags": ["NOUN", "VERB", "NOUN"]}]
    If no output file is specified, the lines will be printed to stdout.
    The data will be formatted in the "simple training style" and can be
    read in and used to update the tagger. See the spaCy documentation:
    https://spacy.io/usage/training#example-train-tagger
    """
    log("RECIPE: Starting recipe pos.gold-to-tags", locals())
    DB = connect()
    if not dataset in DB:
        prints("Can't find dataset '{}'".format(dataset), exits=1, error=True)
    examples = DB.get_dataset(dataset)
    examples = [eg for eg in examples if eg['answer'] == 'accept']
    examples = merge_tags(examples)
    tags_data = []
    skipped_count = 0
    log("RECIPE: Iterating over annotations in the dataset")
    for eg in examples:
        tags = []
        span_tokens = {}
        for span in eg.get('spans', []):
            token_start = span['token_start']
            if (token_start+1) != span['token_end']:
                # skip example if annotated span contains more than one token
                skipped_count += 1
                log("RECIPE: Skipping example with invalid span. POS tag "
                    "spans can't contain multiple tokens (start {}, end {})"
                    .format(token_start, span['token_end']), span)
            else:
                span_tokens[token_start] = span
        if span_tokens:
            for token in eg.get('tokens', []):
                tag = '-'  # default tag: no tag assigned
                if token['id'] in span_tokens:
                    tag = span_tokens[token['id']]['label']
                tags.append(tag)
            if tags:
                tags_entry = [eg['text'], {'tags': tags}]
                tags_data.append(tags_entry)
                if not output_file:
                    print(ujson.dumps(tags_entry,
                                      escape_forward_slashes=False))
    if output_file:
        log("RECIPE: Generated {} examples".format(len(tags_data)))
        write_jsonl(output_file, tags_data)
        prints("Exported {} examples (skipped {} containing invalid spans)"
               .format(len(tags_data), skipped_count), output_file)
