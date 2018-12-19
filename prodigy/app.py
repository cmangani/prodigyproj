# coding: utf8
from __future__ import unicode_literals

import hug
from hug_middleware_cors import CORSMiddleware
from pathlib import Path
import os

from . import about
from .util import prints, log, FLAG_TRASH_SESSION, FLAG_EXPORT_SESSION


CONTROLLER = None
CONFIG = {}
HUG_API = hug.API(__name__)


def set_controller(controller, config):
    """Prepare a controller/config for hosting the Hug prodigy app"""
    global CONTROLLER, CONFIG
    config['view_id'] = controller.view_id
    config['batch_size'] = controller.batch_size
    config['version'] = about.__version__
    instructions = config.get('instructions')
    if instructions:
        help_path = Path(instructions)
        if not help_path.is_file():
            prints("Can't read instructions", help_path, error=True, exits=1)
        with help_path.open('r', encoding='utf8') as f:
            config['instructions'] = f.read()
    for setting in ['db_settings', 'api_keys']:
        if setting in config:
            config.pop(setting)
    CONFIG = config
    CONTROLLER = controller


def server(controller, config):
    """Serve the Prodigy REST API.

    controller (prodigy.core.Controller): The initialized controller.
    config (dict): Configuration settings, e.g. via a prodigy.json or recipe.
    """
    global HUG_API
    set_controller(controller, config)

    from waitress.server import create_server
    if config.get('cors', True) is not False:
        HUG_API.http.add_middleware(CORSMiddleware(HUG_API))
    port = os.getenv('PRODIGY_PORT', config.get('port', 8080))
    host = os.getenv('PRODIGY_HOST', config.get('host', 'localhost'))
    server = create_server(__hug_wsgi__, port=port, host=host,  # noqa: F821
                           channel_timeout=300, expose_tracebacks=True,
                           threads=1)
    prints('Starting the web server at http://{}:{} ...'.format(host, port),
           'Open the app in your browser and start annotating!')
    server.run()
    controller.save()


@hug.response_middleware()
def process_data(request, response, resource):
    # required to make sure browsers never cache the response (e.g. IE 11)
    response.set_header('Cache-control', 'no-cache')


@hug.static('/')
def serve_static():
    return (str(Path(__file__).parent / 'static'),)


@hug.get("/version")
def version():
    return dict(
        name="prodigy",
        description=about.__summary__,
        author=about.__author__,
        author_email=about.__email__,
        url=about.__uri__,
        version=about.__version__,
        license=about.__license__,
    )


@hug.get('/project')
def get_project():
    """Get the meta data and configuration of the current project.

    RETURNS (dict): The configuration parameters and settings.
    """
    log("GET: /project", CONFIG)
    config = CONFIG
    return config


@hug.get('/get_questions')
def get_questions():
    """Get the next batch of tasks to annotate.
    RETURNS (dict): {'tasks': list, 'total': int, 'progress': float}
    """
    log("GET: /get_questions")
    controller = CONTROLLER
    if controller.db and hasattr(controller.db, 'reconnect'):
        controller.db.reconnect()
    tasks = controller.get_questions()
    response = {'tasks': tasks, 'total': controller.total_annotated,
                'progress': controller.progress}
    log("RESPONSE: /get_questions ({} examples)".format(len(tasks)), response)
    if controller.db and hasattr(controller.db, 'close'):
        controller.db.close()
    return response


@hug.post('/get_session_questions')
def get_session_questions(session_id: hug.types.text):
    """Get the next batch of tasks to annotate for a given session_id

    RETURNS (dict): {'tasks': list, 'total': int, 'progress': float, 'session_id': str}
    """
    log("POST: /get_session_questions")
    controller = CONTROLLER
    if controller.db and hasattr(controller.db, 'reconnect'):
        controller.db.reconnect()
    tasks = controller.get_questions(session_id=session_id)
    out_session = session_id if session_id is not None else controller.session_id
    response = {'tasks': tasks, 'total': controller.total_annotated,
                'progress': controller.progress, 'session_id': out_session}
    log("RESPONSE: /get_session_questions ({} examples)".format(len(tasks)), response)
    if controller.db and hasattr(controller.db, 'close'):
        controller.db.close()
    return response


@hug.post('/set_session_aliases')
def set_session_aliases(session_id: hug.types.text, aliases=[]):
    """Set the list of past session_ids to associate with a current session_id. This
    is useful for recipes that require overlap but want to exclude questions an annotator
    has seen before in a previous session for the same task.

    RETURNS (dict): {'session_id': str}
    """
    log("POST: /set_session_aliases")
    controller = CONTROLLER
    controller.set_session_aliases(session_id, aliases)
    response = {'session_id': session_id}
    return response


@hug.post('/end_session')
def end_session(session_id: hug.types.text):
    """Tell the prodigy controller that it can release the resources held for the
    given session_id.
    """
    log("POST: /end_session")
    controller = CONTROLLER
    response = controller.end_session(session_id)
    log("RESPONSE: /end_session ({})".format(response), response)
    return response


@hug.post('/give_answers')
def give_answers(answers=[], session_id=None):
    """Receive annotated answers, e.g. from the web app.

    answers (list): A list of task dictionaries with an added `"answer"` key.
    session_id (str): The session id string that points to a dataset
    RETURNS (dict): {'progress': float}
    """
    log("POST: /give_answers (received {})".format(len(answers)), answers)
    controller = CONTROLLER
    if controller.db and hasattr(controller.db, 'reconnect'):
        controller.db.reconnect()
    controller.receive_answers(answers, session_id=session_id)
    response = {'progress': controller.progress}
    log("RESPONSE: /give_answers", response)
    if controller.db and hasattr(controller.db, 'close'):
        controller.db.close()
    return response


@hug.post('/trash_session')
def trash_session(session_id=None):
    """Remove a session dataset and the examples associated with it.

    session_id (str): The session id string that points to a dataset
    RETURNS (dict): {'total': int, 'trash_file': str}
    """
    log("POST: /trash_session (session {})".format(session_id))
    config = CONFIG
    controller = CONTROLLER
    if not config.get('config', {}).get(FLAG_TRASH_SESSION):
        log("POST: /trash_session not enabled, not deleting anything")
        return None
    if controller.db and hasattr(controller.db, 'reconnect'):
        controller.db.reconnect()
    count, trash_file = controller.trash_session(session_id)
    response = {'total': count, 'trash_file': trash_file}
    log("RESPONSE: /trash_session", response)
    if controller.db and hasattr(controller.db, 'close'):
        controller.db.close()
    return response


@hug.post('/export_session')
def export_session(session_id=None):
    """Export the collected examples for a given session and return them as an array
    of JSON objects that can be written as JSONL.

    session_id (str): The session id string that points to a dataset
    RETURNS (list): An array of examples that matched the given session.
    """
    log("POST: /export_session (session {})".format(session_id), None)
    controller = CONTROLLER
    config = CONFIG
    if not config.get('config', {}).get(FLAG_EXPORT_SESSION):
        log("POST: /export_session not enabled, not exporting anything")
        return None
    if controller.db and hasattr(controller.db, "reconnect"):
        controller.db.reconnect()
    response = controller.export_session(session_id)
    log("RESPONSE: /export_session", response)
    if controller.db and hasattr(controller.db, "close"):
        controller.db.close()
    return response


if __name__ == '__main__':
    import plac
    plac.call(server)
