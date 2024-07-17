import sqlite3

import click
from flask import current_app, g

def get_db():
    """
    Establish a connection to the database or view
    it if it already exists
    """
    if 'db' not in g:
        g.db = sqlite3.connect(
            current_app.config['DATABASE'],
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db

def close_db(e = None):
    """Close connection to the database if it exists"""
    db = g.pop('db', None)

    if db is not None:
        db.close()

def init_db():
    """Initialize the database"""
    db = get_db()

    with current_app.open_resource('database/schema.sql') as f:
        db.executescript(f.read().decode('utf8'))

@click.command('init-db')
def init_db_command():
    """Clear the existing data and create new tables"""
    init_db()
    click.echo('Initialized the database.')


def init_app(app):
    """
    Register the close_db and init_db_command
    functions with an application instance

    Parameters: app - the application instance
    """
    app.teardown_appcontext(close_db)
    app.cli.add_command(init_db_command)