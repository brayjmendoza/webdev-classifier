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

def clear_cancer():
    """Clear exisiting iris table and recreate it"""
    db = get_db()

    with current_app.open_resource('database/sql/cancer.sql') as f:
        db.executescript(f.read().decode('utf8'))


def clean_files(classify = None, model = None):
    """
    Cleanup files that may have been created while running

    Examples: new models, new plots to visualize new models

    Parameters:
    classify - specifies for what object to clean the files for (e.g. clean only iris files)
    model - specifies for what model to clean the files for (e.g. only KNN model)
    """

    # Remove new models
    if classify:
        directory = f'classifier/models/{classify}'
    else:
        directory = 'classifier/models'

    for file in os.walk(directory):
        if file[2] == []:
            continue
        
        prefix = file[0]
        for i in range(len(file[2])):
            if model:
                if model in file[2][i] and 'new' in file[2][i]:
                    os.remove(f'{prefix}/{file[2][i]}')
            else:
                if 'new' in file[2][i]:
                    os.remove(f'{prefix}/{file[2][i]}')

    # Remove plots for predictions and new models
    img_dir = list(os.walk('classifier/static/img'))
    img_prefix = img_dir[0][0] # Get directory path
    for file in img_dir[0][2]:
        if classify and model:
            if classify in file and model in file:
                if 'new' in file or 'this' in file:
                    os.remove(f'{img_prefix}/{file}')
        elif classify:
            if classify in file:
                if 'new' in file or 'this' in file:
                    os.remove(f'{img_prefix}/{file}')
        else:
            if 'new' in file or 'this' in file:
                os.remove(f'{img_prefix}/{file}')

@click.command('reset-iris')
def reset_iris_command():
    """
    Clear iris database and iris files as if the
    iris pages/models have never been touched
    """
    # Clear iris table in database
    clear_iris()
    click.echo('Reset iris table.')

    # Delete any retrained iris models
    models = list(os.walk('classifier/models/iris'))[0][2]
    for model in models:
        if 'new' in model:
            os.remove(f'classifier/models/iris/{model}')
    click.echo("Removed retrained iris models.")

    # Delete any iris model visualization aside from default plots
    plots = list(os.walk('classifier/static/img'))[0][2]
    for plot in plots:
        if 'iris' in plot and 'new' in plot:
            os.remove(f'classifier/static/img/{plot}')
        if 'iris' in plot and 'this' in plot:
            os.remove(f'classifier/static/img/{plot}')
    click.echo("Removed visualization plots.")


@click.command('reset-cancer')
def reset_cancer_command():
    """
    Clear cancer database and cancer files as if the
    cancer pages/models have never been touched
    """
    # Clear cancer table in database
    clear_cancer()
    click.echo('Reset cancer table.')

    # Delete any retrained cancer models
    models = list(os.walk('classifier/models/cancer'))[0][2]
    for model in models:
        if 'new' in model:
            os.remove(f'classifier/models/cancer/{model}')
    click.echo("Removed visualization plots.")

    # Delete any cancer model visualizations aside from default plots
    plots = list(os.walk('classifier/static/img'))[0][2]
    for plot in plots:
        if 'cancer' in plot and 'new' in plot:
            os.remove(f'classifier/static/img/{plot}')
        if 'cancer' in plot and 'this' in plot:
            os.remove(f'classifier/static/img/{plot}')
    click.echo("Removed retrained model plots.")

@click.command('clean')
def clean_command():
    """
    Delete all files created when the app
    was being used
    """
    clean_files()
    click.echo('Cleaned files.')

@click.command('reset-app')
def reset_app_command():
    """
    Re-initialize the database and cleanup files as if the
    app has never been used before
    """
    clean_files()
    init_db()

    click.echo('Reset app.')

def register_commands(app):
    """Register commands to app"""
    app.cli.add_command(reset_iris_command)
    app.cli.add_command(reset_cancer_command)
    app.cli.add_command(clean_command)
    app.cli.add_command(reset_app_command)