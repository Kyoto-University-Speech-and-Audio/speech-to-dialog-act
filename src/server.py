import argparse
import json
import os

import flask
import tensorflow as tf
from flask import request, abort
from flask.json import jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

from .trainers.trainer import Trainer
from . import configs
from .utils import utils

app = flask.Flask(__name__)
CORS(app)
CONFIG_FILES = {
    'ja': {
        'config': 'attention_aps_sps_word',
        'load': 'pretrained_models/attention_aps_sps_word/csp.epoch6.ckpt'
    }
}

trainers = {}

TMP_FOLDER = "./tmp"


def init():
    global trainers
    for id in CONFIG_FILES:
        json_file = open('model_configs/%s.json' % CONFIG_FILES[id]['config']).read()
        json_dict = json.loads(json_file)
        input_cls = utils.get_batched_input_class(json_dict.get("dataset", "default"))
        model_cls = utils.get_model_class(json_dict.get("model"))

        hparams = utils.create_hparams(None, model_cls, config=CONFIG_FILES[id]['config'])
        hparams.hcopy_path = configs.HCOPY_PATH
        hparams.hcopy_config = configs.HCOPY_CONFIG_PATH

        hparams.input_path = os.path.join("tmp", "input.tmp")

        tf.reset_default_graph()
        graph = tf.Graph()
        mode = tf.estimator.ModeKeys.PREDICT

        with graph.as_default():
            trainer = Trainer(hparams, model_cls, input_cls, mode, graph)
            trainer.build_model()

            sess = tf.Session(graph=graph)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, CONFIG_FILES[id]['load'])
            trainer.init(sess)
            trainer.sess = sess

        trainers[id] = trainer


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'webm'}


@app.route("/infer", methods=["POST"])
def infer():
    if 'file' not in request.files or not allowed_file(request.files['file'].filename):
        abort(400)

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(TMP_FOLDER, filename)
    file.save(filepath)

    trainer = trainers[request.data.get('lang', 'ja')]

    input_path = os.path.join("tmp", "input.tmp")
    with open(input_path, "w") as f:
        f.write("sound\n")
        f.write("%s\n" % filepath)

    with trainer.graph.as_default():
        ret = trainer.infer()

    return jsonify([{
        'text': r
    } for r in ret])


if __name__ == "__main__":
    init()
    app.run(threaded=True)
