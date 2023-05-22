#!/usr/bin/python
#!pip3 install pandas

from flask import Flask
from flask_restx import Api, Resource, fields
import joblib
from model_deployment import predict_proba

app = Flask(__name__)

api = Api(
    app, 
    version='1.0', 
    title='Movie Genre Prediction API',
    description='Movie Genre Prediction API')

ns = api.namespace('predict', 
     description='Movie Genre Classifier')
   
parser = api.parser()

parser.add_argument(
    'plot', 
    type=str, 
    required=True, 
    help='Plot to be analyzed', 
    location='args')

resource_fields = api.model('Resource', {
    'result': fields.String,
})

@ns.route('/')
class GenreApi(Resource):

    @api.doc(parser=parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        
        return {
         "result": predict_proba(args['plot'])
        }, 200
    
    
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
