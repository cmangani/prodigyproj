# coding: utf8
from __future__ import unicode_literals, print_function

import plac
import ujson
import platform
from pathlib import Path

from prodigy import about
from prodigy.components.db import connect
from prodigy.components.loaders import get_stream, get_loader
from prodigy.components.filters import filter_inputs
from prodigy.core import recipe_args
from prodigy.util import get_timestamp_session_id, set_hashes, write_jsonl, log
from prodigy.util import get_entry_points, prints, print_stats, PRODIGY_HOME

# See
# https://stackoverflow.com/questions/14207708/ioerror-errno-32-broken-pipe-python
try:
    from signal import signal, SIGPIPE, SIG_DFL
    signal(SIGPIPE, SIG_DFL)
except ImportError:  # Windows
    pass

# when the recipe functions are imported, they're automatically added
user_recipes = get_entry_points('prodigy_recipes')
if user_recipes:
    log("CLI: Added {} recipe(s) via entry points".format(len(user_recipes)))


@plac.annotations(
    set_id=recipe_args['dataset'],
    description=("Dataset description", "positional", None, str),
    author=("Dataset author", "option", "a", str))
def dataset(set_id, description=None, author=None):
    """
    Create a new Prodigy dataset. This lets you assign meta information,
    like a description, and will add the new set to the database. In order to
    collect annotations and save the results, Prodigy expects a dataset ID to
    exist in the database.
    """
    DB = connect()
    if set_id in DB:
        prints("'{}' already exists in database {}."
               .format(set_id, DB.db_name), exits=True, error=True)
    meta = {'description': description, 'author': author}
    created = DB.add_dataset(set_id, meta)
    if not created:
        prints("Couldn't add {} to database {}.".format(set_id, DB.db_name),
               exits=1, error=True)
    prints("Successfully added '{}' to database {}."
           .format(set_id, DB.db_name))


@plac.annotations(
    source=recipe_args['source'],
    api=recipe_args['api'],
    loader=recipe_args['loader'],
    from_dataset=("Stream from a dataset", "flag", "D"),
    exclude=recipe_args['exclude'])
def pipe(source=None, api=None, loader=None, from_dataset=False, exclude=None):
    """
    Load examples from an input source, and print them as newline-delimited
    JSON. This makes it easy to filter the stream with command-line utilities
    such as `grep`. It's also often useful to inspect the stream, by piping to
    `less`.
    """
    DB = connect()
    if from_dataset:
        stream = DB.get_dataset(source)
    else:
        stream = get_stream(source, api, loader)
        stream = (set_hashes(eg) for eg in stream)
    if exclude:
        log("RECIPE: Excluding tasks from datasets: {}"
            .format(', '.join(exclude)))
        exclude_hashes = DB.get_input_hashes(*exclude)
        stream = filter_inputs(stream, exclude_hashes)
    try:
        for eg in stream:
            print(ujson.dumps(eg, escape_forward_slashes=False))
    except KeyboardInterrupt:
        pass


@plac.annotations(
    set_id=recipe_args['dataset'],
    list_datasets=("Print list of all datasets", "flag", "l", bool),
    list_sessions=("Print list of all sessions", "flag", "ls", bool),
    no_format=("Don't pretty-print results", "flag", "NF", bool))
def stats(set_id=None, list_datasets=False, list_sessions=False,
          no_format=False):
    """
    Print Prodigy and database statistics. Specifying a dataset ID will show
    detailed stats for the set.
    """
    DB = connect()
    prodigy_stats = {'version': about.__version__,
                     'location': str(Path(__file__).parent),
                     'prodigy_home': PRODIGY_HOME,
                     'platform': platform.platform(),
                     'python_version': platform.python_version(),
                     'database_name': DB.db_name,
                     'database_id': DB.db_id,
                     'total_datasets': len(DB.datasets),
                     'total_sessions': len(DB.sessions)}
    print_stats('Prodigy stats', prodigy_stats, no_format=no_format)
    if (list_datasets or list_sessions) and len(DB.datasets):
        print_stats('Datasets', DB.datasets, no_format, False)
    if list_sessions and len(DB.sessions):
        print_stats('Sessions', DB.sessions, no_format, False)
    if set_id:
        if set_id not in DB:
            prints("Can't find '{}' in database {}."
                   .format(set_id, DB.db_name), exits=1, error=True)
        examples = DB.get_dataset(set_id)
        meta = DB.get_meta(set_id)
        decisions = {'accept': 0, 'reject': 0, 'ignore': 0}
        for eg in examples:
            if 'answer' in eg:
                decisions[eg['answer']] += 1
            elif 'spans' in eg:
                for span in eg['spans']:
                    if 'answer' in span:
                        decisions[span['answer']] += 1
        dataset_stats = {'dataset': set_id,
                         'created': meta.get('created'),
                         'description': meta.get('description'),
                         'author': meta.get('author'),
                         'annotations': len(examples),
                         'accept': decisions['accept'],
                         'reject': decisions['reject'],
                         'ignore': decisions['ignore']}
        print_stats("Dataset '{}'".format(set_id), dataset_stats,
                    no_format=no_format)


@plac.annotations(set_id=recipe_args['dataset'])
def drop(set_id):
    """
    Remove a dataset. Can't be undone. For a list of all dataset and session
    IDs in the database, use `prodigy stats -ls`.
    """
    DB = connect()
    if set_id not in DB:
        prints("Can't find '{}' in database {}.".format(set_id, DB.db_name),
               exits=1, error=True)
    dropped = DB.drop_dataset(set_id)
    if not dropped:
        prints("Can't remove '{}' from database {}."
               .format(set_id, DB.db_name), exits=1, error=True)
    prints("Removed '{}' from database {}.".format(set_id, DB.db_name),
           exits=1)


@plac.annotations(
    set_id=recipe_args['dataset'],
    in_file=("Path to input annotation file", "positional", None, Path),
    loader=recipe_args['loader'],
    answer=("Set this answer key if none is present", "option", "a", str),
    overwrite=("Overwrite existing answers", "flag", "o", bool),
    dry=("Perform a dry run", "flag", "D", bool))
def db_in(set_id, in_file, loader=None, answer='accept', overwrite=False,
          dry=False):
    """
    Import annotations to the database. Supports all formats loadable by
    Prodigy.
    """
    DB = connect()
    if not in_file.exists() or not in_file.is_file():
        prints("Not a valid input file.", in_file, exits=1, error=True)
    if set_id not in DB:
        prints("Can't find '{}' in database {}.".format(set_id, DB.db_name),
               "Maybe you misspelled the name or forgot to add the dataset "
               "using the `dataset` command?", exits=1, error=True)
    loader = get_loader(loader, file_path=in_file)
    annotations = loader(in_file)
    annotations = [set_hashes(eg) for eg in annotations]
    added_answers = 0
    for task in annotations:
        if 'answer' not in task or overwrite:
            task['answer'] = answer
            added_answers += 1
    session_id = get_timestamp_session_id()
    if not dry:
        DB.add_dataset(session_id, session=True)
        DB.add_examples(annotations, datasets=[set_id, session_id])
    prints("Imported {} annotations for '{}' to database {}"
           .format(len(annotations), set_id, DB.db_name),
           "Added '{}' answer to {} annotations".format(answer, added_answers),
           "Session ID: {}".format(session_id))


@plac.annotations(
    set_id=recipe_args['dataset'],
    out_dir=("Path to output directory", "positional", None, Path),
    answer=("Only export annotations with this answer", "option", "a", str),
    flagged_only=("Only export flagged annotations", "flag", "F", bool),
    dry=("Perform a dry run", "flag", "D", bool))
def db_out(set_id, out_dir=None, answer=None, flagged_only=False, dry=False):
    """
    Export annotations from the database. Files will be exported in
    Prodigy's JSONL format.
    """
    DB = connect()
    if set_id not in DB:
        prints("Can't find '{}' in database {}.".format(set_id, DB.db_name),
               exits=1, error=True)
    examples = DB.get_dataset(set_id)
    if flagged_only:
        examples = [eg for eg in examples if eg.get('flagged')]
    if answer:
        examples = [eg for eg in examples if eg.get('answer') == answer]
    if out_dir is None:
        for eg in examples:
            print(ujson.dumps(eg, escape_forward_slashes=False))
    else:
        if not out_dir.exists():
            out_dir.mkdir()
        out_file = out_dir / '{}.jsonl'.format(set_id)
        if not dry:
            write_jsonl(out_file, examples)
        prints("Exported {} annotations for '{}' from database {}"
               .format(len(examples), set_id, DB.db_name),
               out_file.resolve())


if __name__ == '__main__':
    from prodigy.core import get_recipe, list_recipes
    from prodigy.app import server
    import sys

    commands = {'dataset': dataset, 'drop': drop, 'stats': stats, 'pipe': pipe,
                'db-in': db_in, 'db-out': db_out}

    help_args = ('--help', '-h', 'help')
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] in help_args):
        recipes = list_recipes()
        ner_recipes = [r for r in recipes if r.startswith('ner')]
        textcat_recipes = [r for r in recipes if r.startswith('textcat')]
        other_recipes = [r for r in recipes if not r.startswith('ner')
                         and not r.startswith('textcat')]
        prints("Available recipes:", ', '.join(ner_recipes), '\n',
               ', '.join(textcat_recipes), '\n', ', '.join(other_recipes))
        prints("Available commands:", ', '.join(commands.keys()), exits=1)

    command = sys.argv.pop(1)
    sys.argv[0] = 'prodigy {}'.format(command)
    args = sys.argv[1:]
    if command in commands:
        plac.call(commands[command], arglist=args, eager=False)
    else:
        path = None
        if '-F' in args:
            path = args.pop(args.index('-F') + 1)
            args.pop(args.index('-F'))
        recipe = get_recipe(command, path=path)
        if recipe:
            controller = recipe(*args, use_plac=True)
            if hasattr(controller, 'config'):  # hacky controller check :(
                server(controller, controller.config)
        else:
            if path is not None:
                recipe_path = Path(path)
                if not recipe_path.is_file():
                    prints("Invalid recipe file path", recipe_path.resolve(),
                           exits=1, error=True)
                prints("Can't import recipe '{}'.".format(command),
                       recipe_path.resolve(), exits=1, error=True)
            filtered = [r for r in list_recipes() if command in r]
            similar = ', '.join(filtered)
            prints("Can't find recipe or command '{}'.".format(command),
                   "Similar recipes: {}"
                   .format(similar) if len(filtered) else '',
                   error=True, exits=1)
