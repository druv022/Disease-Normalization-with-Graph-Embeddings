from nerds.doc.document import AnnotatedDocument
from flask import jsonify


class Response(object):

    def _check_input(self, annotated_document):
        if not annotated_document:
            raise ValueError("No document to process, quiting")
        if not isinstance(annotated_document, AnnotatedDocument):
            raise TypeError("Expecting a document, got '{}' instead".format(type(annotated_document)))

    def _build_results_message(self, annotated_document):
        message = [{'text': annotated_document.plain_text_}]
        entities = []
        for annotation in annotated_document.annotations:
            entities.append({
                'id': annotation.identifier,
                'text': annotation.text,
                'type': annotation.label,
                'start': annotation.offset[0],
                'end': annotation.offset[1]
            })
        relations = []
        for relation in annotated_document.relations:
            relations.append({
                'id': relation.identifier,
                'type': relation.label,
                'source': relation.source_id,
                'target': relation.target_id
            })
        normalizations = []
        for norm in annotated_document.normalizations:
            normalizations.append({
                'argument_id': norm.argument_id,
                'resource_id': norm.resource_id,
                'external_id': norm.external_id,
                'preferred_term': norm.preferred_term
            })
        if entities:
            message.append({'entities': entities})
        if relations:
            message.append({'relations': relations})
        if normalizations:
            message.append({'normalizations': normalizations})
        return message

    def build(self, annotated_document):
        self._check_input(annotated_document)
        message = self._build_results_message(annotated_document)
        response = jsonify({'response': message})
        response.status_code = 200
        return response

    def unsupported_file_type_error(self, filename):
        message = {'status': 400, 'message': "Unsupported file type: '{}'".format(filename)}
        response = jsonify(message)
        response.status_code = 400
        return response

    def missing_parameter_error(self, parameter):
        message = {'status': 400, 'message': "Missing parameter '{}'".format(parameter)}
        response = jsonify(message)
        response.status_code = 400
        return response
