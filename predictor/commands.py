import click
import os

def clean_files():
    """
    Cleanup files that may have been created while running

    Examples: new models, new plots to visualize new models
    """

    # Remove new models
    for file in os.walk('predictor/models'):
        if file[2] == []:
            continue
        else:
            prefix = file[0]
            for i in range(len(file[2])):
                if 'new' in file[2][i]:
                    os.remove(f'{prefix}/{file[2][i]}')

    # Remove plots for new models
    img_dir = list(os.walk('predictor/static/img'))
    img_prefix = img_dir[0][0] # Get directory path
    for file in img_dir[0][2]:
        if 'new' in file:
            print(file)
            os.remove(f'{img_prefix}/{file}')

@click.command('clean')
def clean_command():
    clean_files()
    click.echo('Cleaned files.')


def register_commands(app):
    """Register commands to app"""
    app.cli.add_command(clean_command)