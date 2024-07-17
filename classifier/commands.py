import click
import os

from flask import current_app
from classifier.database.db import get_db, init_db

########################
## DATABASE FUNCTIONS
########################
def clear_iris():
    """Clear exisiting iris table and recreate it"""
    db = get_db()

    with current_app.open_resource('database/sql/iris.sql') as f:
        db.executescript(f.read().decode('utf8'))


def clean_files():
    """
    Cleanup files that may have been created while running

    Examples: new models, new plots to visualize new models
    """

    # Remove new models
    for file in os.walk('classifier/models'):
        if file[2] == []:
            continue
        else:
            prefix = file[0]
            for i in range(len(file[2])):
                if 'new' in file[2][i]:
                    os.remove(f'{prefix}/{file[2][i]}')

    # Remove plots for new models
    img_dir = list(os.walk('classifier/static/img'))
    img_prefix = img_dir[0][0] # Get directory path
    for file in img_dir[0][2]:
        if 'new' in file:
            os.remove(f'{img_prefix}/{file}')


@click.command('reset-iris')
def reset_iris_command():
    clear_iris()
    click.echo('Reset iris table.')

    models = list(os.walk('classifier/models/iris'))[0][2]
    for model in models:
        if 'new' in model:
            os.remove(f'classifier/models/iris/{model}')
    click.echo("Removed retrained iris models.")

    plots = list(os.walk('classifier/static/img'))[0][2]
    for plot in plots:
        if 'iris' and 'new' in plot:
            os.remove(f'classifier/static/img/{plot}')
    click.echo("Removed retrained model plots.")

@click.command('clean')
def clean_command():
    clean_files()
    click.echo('Cleaned files.')

@click.command('reset-app')
def reset_app_command():
    """
    Clear databases and cleanup files as if the
    app has never been used before
    """
    clean_files()
    init_db()

    click.echo('Reset app.')

def register_commands(app):
    """Register commands to app"""
    app.cli.add_command(reset_iris_command)
    app.cli.add_command(clean_command)
    app.cli.add_command(reset_app_command)