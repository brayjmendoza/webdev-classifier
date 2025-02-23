import os

from flask import Flask, render_template
from flask_socketio import SocketIO


socketio = SocketIO()

def create_app(test_config = None):
    # Create and configure app
    app = Flask(__name__, instance_relative_config = True)
    app.config.from_mapping(
        SECRET_KEY = 'dev',
        DATABASE = os.path.join(app.instance_path, 'classifier.sqlite')
    )

    if test_config is None:
        app.config.from_pyfile('config.py', silent = True)
    else:
        app.config.from_mapping(test_config)

    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize socketio to allow for loading messages
    socketio.init_app(app)
    
    # Register database functions/commands
    from classifier.database import db
    db.init_app(app)
    
    # Register classification blueprints
    from classifier.iris.routes import iris_bp
    from classifier.cancer.routes import cancer_bp
    app.register_blueprint(iris_bp, url_prefix='/iris')
    app.register_blueprint(cancer_bp, url_prefix='/cancer')

    # Register additional utility commands
    from . import commands
    commands.register_commands(app)

    @app.route('/')
    def index():
        """Render homepage"""
        return render_template('index.html')

    return app