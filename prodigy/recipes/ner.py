# coding: utf8
from __future__ import unicode_literals

import random
import mmh3
import json
import spacy
import spacy.gold
import spacy.vocab
import spacy.tokens
import copy
import ujson
from pathlib import Path

from .compare import get_questions as get_compare_questions
from ..models.ner import EntityRecognizer, merge_spans, guess_batch_size
from ..models.matcher import PatternMatcher
from ..components import printers
from ..components.db import connect
from ..components.preprocess import split_sentences, add_tokens
from ..components.sorters import prefer_uncertain
from ..components.loaders import get_stream
from ..core import recipe, recipe_args
from ..util import split_evals, get_labels_from_ner, get_print, combine_models
from ..util import write_jsonl, export_model_data, set_hashes, log, prints
from ..util import INPUT_HASH_ATTR, TASK_HASH_ATTR


@recipe('ner.match',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        patterns=recipe_args['patterns'],
        exclude=recipe_args['exclude'],
        resume=("Resume from existing dataset and update matcher accordingly",
                "flag", "R", bool))
def match(dataset, spacy_model, patterns, source=None, api=None, loader=None,
          exclude=None, resume=False):
    """
    Suggest phrases that match a given patterns file, and mark whether they
    are examples of the entity you're interested in. The patterns file can
    include exact strings, regular expressions, or token patterns for use with
    spaCy's `Matcher` class.
    """
    log("RECIPE: Starting recipe ner.match", locals())
    DB = connect()
    # Create the model, using a pre-trained spaCy model.
    model = PatternMatcher(spacy.load(spacy_model)).from_disk(patterns)
    log("RECIPE: Created PatternMatcher using model {}".format(spacy_model))
    if resume and dataset is not None and dataset in DB:
        existing = DB.get_dataset(dataset)
        log("RECIPE: Updating PatternMatcher with {} examples from dataset {}"
            .format(len(existing), dataset))
        model.update(existing)
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    return {
        'view_id': 'ner',
        'dataset': dataset,
        'stream': (eg for _, eg in model(stream)),
        'exclude': exclude
    }


@recipe('ner.teach',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label_set'],
        patterns=recipe_args['patterns'],
        exclude=recipe_args['exclude'],
        unsegmented=recipe_args['unsegmented'])
def teach(dataset, spacy_model, source=None, api=None, loader=None,
          label=None, patterns=None, exclude=None, unsegmented=False):
    """
    Collect the best possible training data for a named entity recognition
    model with the model in the loop. Based on your annotations, Prodigy will
    decide which questions to ask next.
    """
    log("RECIPE: Starting recipe ner.teach", locals())
    # Initialize the stream, and ensure that hashes are correct, and examples
    # are deduplicated.
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    # Create the model, using a pre-trained spaCy model.
    nlp = spacy.load(spacy_model)
    log("RECIPE: Creating EntityRecognizer using model {}".format(spacy_model))
    model = EntityRecognizer(nlp, label=label)
    if label is not None and patterns is None:
        log("RECIPE: Making sure all labels are in the model", label)
        for l in label:
            if not model.has_label(l):
                prints("Can't find label '{}' in model {}"
                       .format(l, spacy_model),
                       "ner.teach will only show entities with one of the "
                       "specified labels. If a label is not available in the "
                       "model, Prodigy won't be able to propose entities for "
                       "annotation. To add a new label, you can specify a "
                       "patterns file containing examples of the new entity "
                       "as the --patterns argument or pre-train your model "
                       "with examples of the new entity and load it back in.",
                       error=True, exits=1)
    if patterns is None:
        predict = model
        update = model.update
    else:
        matcher = PatternMatcher(model.nlp).from_disk(patterns)
        log("RECIPE: Created PatternMatcher and loaded in patterns", patterns)
        # Combine the NER model with the PatternMatcher to annotate both
        # match results and predictions, and update both models.
        predict, update = combine_models(model, matcher)
    # Split the stream into sentences
    if not unsegmented:
        stream = split_sentences(model.orig_nlp, stream)
    # Return components, to construct Controller
    return {
        'view_id': 'ner',
        'dataset': dataset,
        'stream': prefer_uncertain(predict(stream)),
        'update': update,  # callback to update the model in-place
        'exclude': exclude,
        'config': {'lang': model.nlp.lang,
                   'label': (', '.join(label)) if label is not None else 'all'}
    }


@recipe('ner.manual',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label_set'],
        exclude=recipe_args['exclude'])
def manual(dataset, spacy_model, source=None, api=None, loader=None,
           label=None, exclude=None):
    """
    Mark spans by token. Requires only a tokenizer and no entity recognizer,
    and doesn't do any active learning.
    """
    log("RECIPE: Starting recipe ner.manual", locals())
    nlp = spacy.load(spacy_model)
    log("RECIPE: Loaded model {}".format(spacy_model))
    # Get the label set from the `label` argument, which is either a
    # comma-separated list or a path to a text file. If labels is None, check
    # if labels are present in the model.
    labels = label
    if not labels:
        labels = get_labels_from_ner(nlp)
        print("Using {} labels from model: {}"
              .format(len(labels), ', '.join(labels)))
    log("RECIPE: Annotating with {} labels".format(len(labels)), labels)
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    # Tokenize the text and add a "tokens" key to the annotation task. This is
    # required to allow entity selection based on token boundaries.
    stream = add_tokens(nlp, stream)

    return {
        'view_id': 'ner_manual',
        'dataset': dataset,
        'stream': stream,
        'exclude': exclude,
        'config': {'lang': nlp.lang, 'labels': labels}
    }


@recipe('ner.make-gold',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label_set'],
        exclude=recipe_args['exclude'],
        unsegmented=recipe_args['unsegmented'])
def make_gold(dataset, spacy_model, source=None, api=None, loader=None,
              label=None, exclude=None, unsegmented=False):
    """
    Create gold data for NER by correcting a model's suggestions.
    """
    log("RECIPE: Starting recipe ner.make-gold", locals())
    nlp = spacy.load(spacy_model)
    log("RECIPE: Loaded model {}".format(spacy_model))
    # Get the label set from the `label` argument, which is either a
    # comma-separated list or a path to a text file. If labels is None, check
    # if labels are present in the model.
    labels = label
    if not labels:
        labels = get_labels_from_ner(nlp)
        print("Using {} labels from model: {}"
              .format(len(labels), ', '.join(labels)))
    log("RECIPE: Annotating with {} labels".format(len(labels)), labels)
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        dedup=True, input_key='text')
    # Split the stream into sentences
    if not unsegmented:
        stream = split_sentences(nlp, stream)
    # Tokenize the stream
    stream = add_tokens(nlp, stream)

    def make_tasks(nlp, stream):
        """Add a 'spans' key to each example, with predicted entities."""
        texts = ((eg['text'], eg) for eg in stream)
        for doc, eg in nlp.pipe(texts, as_tuples=True):
            task = copy.deepcopy(eg)
            spans = []
            for ent in doc.ents:
                if labels and ent.label_ not in labels:
                    continue
                spans.append({
                    'token_start': ent.start,
                    'token_end': ent.end-1,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'text': ent.text,
                    'label': ent.label_,
                    'source': spacy_model,
                    'input_hash': eg[INPUT_HASH_ATTR]
                })
            task['spans'] = spans
            task = set_hashes(task)
            yield task

    return {
        'view_id': 'ner_manual',
        'dataset': dataset,
        'stream': make_tasks(nlp, stream),
        'exclude': exclude,
        'update': None,
        'config': {'lang': nlp.lang, 'labels': labels}
    }


@recipe('ner.eval',
        dataset=recipe_args['dataset'],
        model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label_set'],
        exclude=recipe_args['exclude'],
        whole_text=recipe_args['whole_text'],
        unsegmented=recipe_args['unsegmented'])
def evaluate(dataset, model, source=None, api=None, loader=None, label=None,
             exclude=None, whole_text=False, unsegmented=False):
    """
    Evaluate an NER model and build an evaluation set from a stream.
    """
    log("RECIPE: Starting recipe ner.evaluate", locals())

    model = EntityRecognizer(spacy.load(model), label=label)
    stream = get_stream(source, api=api, loader=loader, rehash=True,
                        input_key='text')
    # Split the stream into sentences
    if not unsegmented:
        stream = split_sentences(model.nlp, stream)

    def get_tasks(model, stream):
        tuples = ((eg['text'], eg) for eg in stream)
        for i, (doc, eg) in enumerate(model.nlp.pipe(tuples, as_tuples=True)):
            ents = [(ent.start_char, ent.end_char, ent.label_)
                    for ent in doc.ents]
            if model.labels:
                ents = [seL for seL in ents if seL[2] in model.labels]

            eg['label'] = 'all correct'
            ents = [{'start': s, 'end': e, 'label': L} for s, e, L in ents]
            if whole_text:
                eg['spans'] = ents
                eg = set_hashes(eg, overwrite=True)
                yield eg
            else:
                for span in ents:
                    task = copy.deepcopy(eg)
                    task['spans'] = [span]
                    task = set_hashes(task, overwrite=True)
                    yield task

    return {
        'view_id': 'classification',
        'dataset': dataset,
        'stream': get_tasks(model, stream),
        'exclude': exclude,
        'config': {'lang': model.nlp.lang}
    }


@recipe('ner.eval-ab',
        dataset=recipe_args['dataset'],
        before_model=recipe_args['spacy_model'],
        after_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label_set'],
        exclude=recipe_args['exclude'],
        unsegmented=recipe_args['unsegmented'])
def ab_evaluate(dataset, before_model, after_model, source=None, api=None,
                loader=None, label=None, exclude=None, unsegmented=False):
    """
    Evaluate a n NER model and build an evaluation set from a stream.
    """
    log("RECIPE: Starting recipe ner.eval-ab", locals())

    def get_task(i, text, ents, name):
        spans = [{'start': s, 'end': e, 'label': L} for s, e, L in ents]
        task = {'id': i, 'input': {'text': text},
                'output': {'text': text, 'spans': spans}}
        task[INPUT_HASH_ATTR] = mmh3.hash(name + str(i))
        task[TASK_HASH_ATTR] = mmh3.hash(name + str(i))
        return task

    def get_tasks(model, stream, name):
        tuples = ((eg['text'], eg) for eg in stream)
        for i, (doc, eg) in enumerate(model.nlp.pipe(tuples, as_tuples=True)):
            ents = [(ent.start_char, ent.end_char, ent.label_)
                    for ent in doc.ents]
            if model.labels:
                ents = [seL for seL in ents if seL[2] in model.labels]
            task = get_task(i, eg['text'], ents, name)
            yield task

    before_model = EntityRecognizer(spacy.load(before_model), label=label)
    after_model = EntityRecognizer(spacy.load(after_model), label=label)
    stream = list(get_stream(source, api=api, loader=loader, rehash=True,
                             dedup=True, input_key='text'))
    if not unsegmented:
        stream = list(split_sentences(before_model.nlp, stream))
    before_stream = list(get_tasks(before_model, stream, 'before'))
    after_stream = list(get_tasks(after_model, stream, 'after'))
    stream = list(get_compare_questions(before_stream, after_stream, True))

    return {
        'view_id': 'compare',
        'dataset': dataset,
        'stream': stream,
        'on_exit': printers.get_compare_printer('Before', 'After'),
        'exclude': exclude
    }


@recipe('ner.batch-train',
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
        unsegmented=recipe_args['unsegmented'],
        no_missing=recipe_args['no_missing'],
        silent=recipe_args['silent'])
def batch_train(dataset, input_model, output_model=None, label='', factor=1,
                dropout=0.2, n_iter=10, batch_size=-1, beam_width=16,
                eval_id=None, eval_split=None, unsegmented=False,
                no_missing=False, silent=False):
    """
    Batch train a Named Entity Recognition model from annotations. Prodigy will
    export the best result to the output directory, and include a JSONL file of
    the training and evaluation examples. You can either supply a dataset ID
    containing the evaluation data, or choose to split off a percentage of
    examples for evaluation.
    """
    log("RECIPE: Starting recipe ner.batch-train", locals())
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
    examples = merge_spans(DB.get_dataset(dataset))
    random.shuffle(examples)
    if 'ner' not in nlp.pipe_names:
        if nlp.vocab.vectors.name is None and nlp.vocab.vectors.size == 0:
            # Workaround for spaCy 2.0.11: Avoids print statement.
            nlp.vocab.vectors.name = ''
        ner = nlp.create_pipe('ner')
        if nlp.vocab.vectors.data.size:
            ner.cfg['pretrained_dims'] = nlp.vocab.vectors.data.shape[1]
    else:
        ner = nlp.get_pipe('ner')
    # make sure all labels are present in the model
    for eg in examples:
        for span in eg.get('spans', []):
            ner.add_label(span['label'])
    for l in label:
        ner.add_label(l)
    if 'ner' not in nlp.pipe_names:
        nlp.add_pipe(ner, last=True)
        nlp.begin_training()
    if eval_id:
        evals = DB.get_dataset(eval_id)
        print_("Loaded {} evaluation examples from '{}'"
               .format(len(evals), eval_id))
    else:
        examples, evals, eval_split = split_evals(examples, eval_split)
        print_("Using {}% of accept/reject examples ({}) for evaluation"
               .format(round(eval_split * 100), len(evals)))
    model = EntityRecognizer(nlp, label=label, no_missing=no_missing)
    if batch_size < 1:
        batch_size = guess_batch_size(len(examples))

    other_pipes = [p for p in nlp.pipe_names if p not in ('ner', 'sbd')]
    if other_pipes:
        disabled = nlp.disable_pipes(*other_pipes)
        log("RECIPE: Temporarily disabled other pipes: {}".format(other_pipes))
    else:
        disabled = None
    log('RECIPE: Initialised EntityRecognizer with model {}'
        .format(input_model), model.nlp.meta)
    if not unsegmented:
        examples = list(split_sentences(model.orig_nlp, examples))
        evals = list(split_sentences(model.orig_nlp, evals))
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
    print_(printers.ner_before(**baseline))
    if len(evals) > 0:
        print_(printers.ner_update_header())
    for i in range(n_iter):
        losses = model.batch_train(examples, batch_size=batch_size,
                                   drop=dropout, beam_width=beam_width)
        stats = model.evaluate(evals)
        if best is None or stats['acc'] > best[0]:
            model_to_bytes = None
            if output_model is not None:
                model_to_bytes = model.to_bytes()
            best = (stats['acc'], stats, model_to_bytes)
        print_(printers.ner_update(i, losses, stats))
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


@recipe('ner.train-curve',
        dataset=recipe_args['dataset'],
        input_model=recipe_args['spacy_model'],
        label=recipe_args['label_set'],
        dropout=recipe_args['dropout'],
        n_iter=recipe_args['n_iter'],
        batch_size=recipe_args['batch_size'],
        beam_width=recipe_args['beam_width'],
        eval_id=recipe_args['eval_id'],
        eval_split=recipe_args['eval_split'],
        unsegmented=recipe_args['unsegmented'],
        no_missing=recipe_args['no_missing'],
        n_samples=recipe_args['n_samples'])
def train_curve(dataset, input_model, label='', dropout=0.2, n_iter=5,
                batch_size=32, beam_width=16, eval_id=None, eval_split=None,
                unsegmented=False, no_missing=False, n_samples=4):
    """
    Batch-train models with different portions of the training examples and
    print the accuracy figures and accuracy improvements.
    """
    log("RECIPE: Starting recipe ner.train-curve", locals())
    factors = [(i + 1) / n_samples for i in range(n_samples)]
    prev_acc = 0
    print("\nStarting with model {}".format(input_model))
    print(printers.trainconf(dropout, n_iter, batch_size, samples=n_samples))
    print(printers.ner_curve_header())
    for factor in factors:
        best_stats = batch_train(dataset, input_model=input_model, label=label,
                                 factor=factor, dropout=dropout,
                                 n_iter=n_iter, batch_size=batch_size,
                                 beam_width=beam_width, eval_id=eval_id,
                                 eval_split=eval_split, no_missing=no_missing,
                                 unsegmented=unsegmented, silent=True)
        print(printers.ner_curve(factor, best_stats, prev_acc))
        prev_acc = best_stats['acc']


@recipe('ner.print-best',
        dataset=recipe_args['dataset'],
        spacy_model=recipe_args['spacy_model'],
        pretty=("Pretty-print output", "flag", "P", bool))
def print_best(dataset, spacy_model, pretty=False):
    """
    Predict the highest-scoring parse for examples in a dataset. Scores are
    calculated using the annotations in the dataset, and the statistical model.
    """
    log("RECIPE: Starting recipe ner.best-parse", locals())
    DB = connect()
    if not dataset in DB:
        prints("Can't find dataset '{}'".format(dataset), exits=1, error=True)
    model = EntityRecognizer(spacy.load(spacy_model))
    log('RECIPE: Initialised EntityRecognizer with model {}'
        .format(spacy_model), model.nlp.meta)
    log("RECIPE: Outputting stream of examples")
    stream = model.make_best(DB.get_dataset(dataset))
    if pretty:
        printers.pretty_print_ner(stream)
    else:
        for eg in stream:
            print(json.dumps(eg))


@recipe('ner.print-stream',
        spacy_model=recipe_args['spacy_model'],
        source=recipe_args['source'],
        api=recipe_args['api'],
        loader=recipe_args['loader'],
        label=recipe_args['label_set'])
def pretty_print_stream(spacy_model, source=None, api=None, loader=None,
                        label=''):
    """
    Pretty print stream output.
    """
    log("RECIPE: Starting recipe ner.print-stream", locals())

    def add_entities(stream, nlp, labels=None):
        for eg in stream:
            doc = nlp(eg['text'])
            ents = [{'start': e.start_char, 'end': e.end_char,
                     'label': e.label_} for e in doc.ents
                    if not labels or e.label_ in labels]
            if ents:
                eg['spans'] = ents
                yield eg

    nlp = spacy.load(spacy_model)
    stream = get_stream(source, api, loader, rehash=True, input_key='text')
    stream = add_entities(stream, nlp, label)
    printers.pretty_print_ner(stream)


@recipe('ner.print-dataset',
        dataset=recipe_args['dataset'])
def pretty_print_dataset(dataset):  # pragma: no cover
    """
    Pretty print dataset.
    """
    log("RECIPE: Starting recipe ner.print-dataset", locals())
    DB = connect()
    if not dataset in DB:
        prints("Can't find dataset '{}'".format(dataset), exits=1, error=True)
    examples = DB.get_dataset(dataset)
    if not examples:
        raise ValueError("Can't load '{}' from database {}"
                         .format(dataset, DB.db_name))
    printers.pretty_print_ner(examples)


@recipe('ner.gold-to-spacy',
        dataset=recipe_args['dataset'],
        output_file=recipe_args['output_file'],
        spacy_model=("Optional spaCy model for tokenization, required for "
                     "BILUO mode", "option", "sm", str),
        biluo=("Encode labelled spans into per-token BILUO tags", "flag", "B",
               bool))
def gold_to_spacy(dataset, output_file=None, spacy_model=None, biluo=False):
    """
    Convert a dataset of gold-standard NER annotations (created with ner.manual
    or ner.make-gold) into training data for spaCy. Will export a JSONL file
    with one entry per line:
        >>> ["I like London", {"entities": [[7, 13, "LOC"]]}]
    If no output file is specified, the lines will be printed to stdout.
    BILUO requires a spaCy model for tokenization, which should be the same
    model used during annotation. The BILUO data will look like this:
        >>> ["I like London", ["O", "O", "U-LOC", "O"]]
    """
    log("RECIPE: Starting recipe ner.gold-to-spacy", locals())
    DB = connect()
    if not dataset in DB:
        prints("Can't find dataset '{}'".format(dataset), exits=1, error=True)
    examples = DB.get_dataset(dataset)
    examples = [eg for eg in examples if eg['answer'] == 'accept']
    if biluo:
        if not spacy_model:
            prints("Exporting annotations in BILUO format requires a spaCy "
                   "model for tokenization.", exits=1, error=True)
        nlp = spacy.load(spacy_model)
    log("RECIPE: Loaded model {}".format(spacy_model))
    log("RECIPE: Iterating over annotations in the dataset")
    annotations = []
    for eg in examples:
        entities = [(span['start'], span['end'], span['label'])
                    for span in eg.get('spans', [])]
        if biluo:
            doc = nlp(eg['text'])
            entities = spacy.gold.biluo_tags_from_offsets(doc, entities)
            annot_entry = [eg['text'], entities]
        else:
            annot_entry = [eg['text'], {'entities': entities}]
        annotations.append(annot_entry)
        if not output_file:
            print(ujson.dumps(annot_entry, escape_forward_slashes=False))
    if output_file:
        log("RECIPE: Generated {} examples".format(len(annotations)))
        write_jsonl(output_file, annotations)
        prints("Exported {} examples".format(len(annotations)), output_file)


@recipe('ner.iob-to-gold',
        input_file=("Path to IOB or IOB2 formatted NER annotations",
                    "positional", None, Path),
        output_file=("Path to write the .jsonl formatted output", "positional",
                     None, Path))
def iob_to_gold(input_file, output_file=None):
    """
    Convert a file with IOB tags into JSONL format for use in Prodigy.

    The input format should have one line per text, with whitespace-delimited
    tokens. Each token should have two or more fields delimited by the |
    character. The first field should be the text, and the last an IOB or IOB2
    formatted NER tag.

    Example (IOB):
    Then|RB|O ,|,|O the|DT|I-MISC International|NNP|I-MISC became|VBD|O polarised|VBN|O

    Example (IOB2):
    Then|RB|O ,|,|O the|DT|B-MISC International|NNP|I-MISC became|VBD|O polarised|VBN|O

    If no output is specified, the output is printed to stdout.
    """
    vocab = spacy.vocab.Vocab()
    golds = []
    with input_file.open('r', encoding='utf8') as f:
        for line in f:
            tokens = [t.split('|') for t in line.split() if t.strip()]
            if not tokens:
                continue
            words = [token[0] for token in tokens]
            iob = [token[-1] for token in tokens]
            doc = spacy.tokens.Doc(vocab, words=words)
            offsets = spacy.gold.offsets_from_biluo_tags(doc, spacy.gold.iob_to_biluo(iob))
            text = doc.text
            spans = [{'text': text[s:e], 'label': L, 'start': s, 'end': e}
                        for s, e, L in offsets]
            task = {'text': doc.text, 'spans': spans, 'no_missing': True}
            task = set_hashes(task)
            golds.append(task)
            if output_file is None:
                print(ujson.dumps(task))
    if output_file is not None:
        log("RECIPE: Converted {} annotations".format(len(golds)))
        write_jsonl(output_file, golds)
        prints("Converted {} annotations".format(len(golds)), output_file)
