import logging
from flask import Flask, Blueprint
from flask_restx import Api, Resource, fields
from time import time

logging.basicConfig(level=logging.INFO)

from aic_summarization_rest_api.mbart_summarizer import MBartSummarizer



model_name = "krotima1/mbart-at2h-cs"
summarizer = MBartSummarizer(model_name, device="cpu")

api_v1 = Blueprint("api", __name__, url_prefix="/api/1")

api = Api(
    api_v1,
    version="1.0",
    title="AIC Summarization API",
    description="A simple text summarization API",
)

ns = api.namespace("summarize", description="generate summaries")

parser = api.parser()
parser.add_argument(
    "sources", type=str, required=True, action='append', help="Source texts to summarize", location="form"
)

apimodel = api.model('SumModel', {
    'summaries': fields.List(fields.String, description="List of summaries, one per each source text"),
    'duration_s': fields.Float(description="Summarization duration in seconds"),
})

@ns.route("/")
class Summarize(Resource):
    """Generate summaries"""

    @api.marshal_with(apimodel, envelope='resource')
    @api.doc(parser=parser)
    def post(self):
        """Generate a summary"""
        args = parser.parse_args()
        st = time()
        summaries = summarizer.summarize_batch(args["sources"])
        duration_s = time()-st
        ret = {"summaries": summaries, "duration_s": duration_s}
        return ret, 200


if __name__ == "__main__":
    app = Flask(__name__)
    app.register_blueprint(api_v1)
    app.run(debug=True)
