import os

from flask import Flask, render_template

def create_app(test_config = None):
    # create and configure app
    app = Flask(__name__, instance_relative_config = True)
    app.config.from_mapping(
        SECRET_KEY = 'dev',
        DATABASE = os.path.join(app.instance_path, 'predictor.sqlite')
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent = True)
    else:
        app.config.from_mapping(test_config)

    # ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    # Register database functions/commands
    from predictor.database import db
    db.init_app(app)
    
    # Register classification blueprints
    from predictor.iris.routes import iris_bp
    app.register_blueprint(iris_bp, url_prefix='/iris')

    # Register additional utility commands
    from . import commands
    commands.register_commands(app)

    @app.route('/')
    def index():
        """Render homepage"""
        return render_template('index.html')

    return app