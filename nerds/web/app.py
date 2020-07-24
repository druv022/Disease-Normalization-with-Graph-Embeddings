from pathlib import Path

from flask import Flask
from flask import request

from nerds.web.input import InputDocumentFile
from nerds.web.load import ModelLoader
from nerds.web.response import Response

app = Flask(__name__)
response = Response()

# load pre-trained models here
# it's a temporal solution that will be replaced by proper model uploading
root_dir = Path(__file__).parent.parent.parent
model_dir = Path(root_dir.joinpath('models'))
loader = ModelLoader()
ner_annotator = loader.load_model(model_dir.joinpath('spacy_ner'))
rel_annotator = loader.load_model(model_dir.joinpath('svm_re'))
nrm_annotator = loader.load_normalizer()


@app.route('/', methods=['GET', 'POST'])
def index():
    return 'This is a Flask application serving as a RESTful API for extracting Drug Safety information ' \
           'from text. Given a piece of text, it extracts drug-related entities (use route "/ner") ' \
           'and/or relations between them (use route "/rel"). It works with POST requests only.'


@app.errorhandler(400)
def missing_parameter_error(parameter):
    return response.missing_parameter_error(parameter)


@app.errorhandler(400)
def unsupported_document_file_error(filename):
    return response.unsupported_file_type_error(filename)


@app.errorhandler(400)
def unsupported_model_file_error(filename):
    return response.unsupported_file_type_error(filename)


def is_supported_document_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'txt'


def is_supported_model_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    return ext == 'zip' or ext == 'tar' or ext == 'gz'


def input_document_file_check():
    if 'file' not in request.files:
        return missing_parameter_error('file')
    file = request.files['file']
    if not file or file.filename == '':
        return missing_parameter_error('file')
    if not is_supported_document_file(file.filename):
        return unsupported_document_file_error(file.filename)
    return None


def input_model_file_check():
    if 'file' not in request.files:
        return missing_parameter_error('file')
    file = request.files['file']
    if not file or file.filename == '':
        return missing_parameter_error('file')
    if not is_supported_model_file(file.filename):
        return unsupported_model_file_error(file.filename)
    return None


@app.route('/ner', methods=['POST'])
def extract_entities():
    err_result = input_document_file_check()
    if err_result:
        return err_result
    # load document
    input_file = InputDocumentFile(request.files['file'])
    # extract entities
    annotated_document = ner_annotator.annotate(input_file.document)
    # build a response
    return response.build(annotated_document)


@app.route('/rel', methods=['POST'])
def extract_relations():
    err_result = input_document_file_check()
    if err_result:
        return err_result
    # load document
    input_file = InputDocumentFile(request.files['file'])
    # extract entities
    annotated_document = ner_annotator.annotate(input_file.document)
    # extract relations
    annotated_document = rel_annotator.annotate(annotated_document)
    # build a response
    return response.build(annotated_document)


@app.route('/nrm', methods=['POST'])
def extract_normalizations():
    err_result = input_document_file_check()
    if err_result:
        return err_result
    # load document
    input_file = InputDocumentFile(request.files['file'])
    # extract entities
    annotated_document = ner_annotator.annotate(input_file.document)
    # normalize entities
    annotated_document = nrm_annotator.annotate(annotated_document)
    # build a response
    return response.build(annotated_document)


@app.route('/all', methods=['POST'])
def extract_all():
    err_result = input_document_file_check()
    if err_result:
        return err_result
    # load document
    input_file = InputDocumentFile(request.files['file'])
    # extract entities
    annotated_document = ner_annotator.annotate(input_file.document)
    # extract relations
    annotated_document = rel_annotator.annotate(annotated_document)
    # normalize entities
    annotated_document = nrm_annotator.annotate(annotated_document)
    # build a response
    return response.build(annotated_document)


@app.route('/model', methods=['POST'])
def upload_model():
    err_result = input_model_file_check()
    if err_result:
        return err_result
    # TODO: upload_model


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5003)
