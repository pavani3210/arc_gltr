import argparse
import os
from flask import Flask, flash, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from flask_cors import CORS, cross_origin
import logging
from backend import AVAILABLE_MODELS

logging.basicConfig(level=logging.INFO)

UPLOAD_FOLDER = '/'
logger = logging.getLogger('')
ALLOWED_EXTENSIONS = set(['txt', 'docx', 'pdf', 'doc'])
CONFIG_FILE_NAME = 'lmf.yml'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

projects={}
class Project:
    def __init__(self, LM, config):
        self.config = config
        self.lm = LM()

def get_all_projects():
    res = {}
    for k in projects.keys():
        res[k] = projects[k].configs
    return res

@app.route('/hello', methods=['GET'])
def test():
    return "hello world"


@app.route('/upload', methods=['POST'])
def fileUpload():
    target=os.path.join(UPLOAD_FOLDER)
    if not os.path.isdir(target):
        os.mkdir(target)
    logger.info("welcome to upload`")
    file = request.files['file'] 
    project = "gpt-2-small"
    res = {}
    if project in projects:
        p = projects[project] # type: Project
        res = p.lm.extract_files(p, file, topk=20)
    res.headers.add("Access-Control-Allow-Origin", "*")
    res.headers.add("Access-Control-Allow-Headers", "*")
    res.headers.add("Access-Control-Allow-Methods", "*")
    return res

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='gpt-2-small')
parser.add_argument("--nodebug", default=True)
parser.add_argument("--address",
                    default="0.0.0.0")  # 0.0.0.0 for nonlocal use
parser.add_argument("--port", default="5001")
parser.add_argument("--nocache", default=False)
parser.add_argument("--dir", type=str, default=os.path.abspath('data'))

parser.add_argument("--no_cors", action='store_true')

args, _ = parser.parse_known_args()
try:
    model = AVAILABLE_MODELS[args.model]
except KeyError:
    print("Model {} not found. Make sure to register it.".format(
        args.model))
    print("Loading GPT-2 instead.")
    model = AVAILABLE_MODELS['gpt-2']
projects[args.model] = Project(model, args.model)

if __name__ == '__main__':
    args = parser.parse_args()
    app.run(port=int(args.port), debug=not args.nodebug, host=args.address)
    
CORS(app, expose_headers='Authorization')