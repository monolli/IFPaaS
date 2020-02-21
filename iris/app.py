'''
This is a minimalistic Iris flower classifier API service.
'''
import os

from flask import Flask, jsonify, request
from flask_restplus import Api, Resource, fields

from iris.predictor import predict_iris

# Create a Flask app
app = Flask(__name__)

# Create a RESTPlus API
api = Api(app,
          version="1.0",
          title="Iris flower classifier",
          description="Classify an Iris flower based on its features.",
          default="classify_iris",
          default_label="Iris flower classifier.",
          license="GNU GPL v3.0",
          license_url="https://raw.githubusercontent.com/monolli/IFPaaS/master/LICENSE")

# Define a JSON model
multiple_model = api.model("Iris dataset sample",
                           {"data": fields.List(fields.List(fields.Float),
                            required=True,
                            example=[[4.8, 3.4, 1.6, 0.2], [6.5, 3.2, 5.1, 2.0]],
                            description="A list containing a list of floats that represent the feature values.")},
                           required=True)

# Define the single parser
single_parser = api.parser()
single_parser.add_argument("sepal_l", type=float, help="Specify the sepal lenght.",
                           location="query", required=True, default=4.8)
single_parser.add_argument("sepal_w", type=float, help="Specify the sepal width.",
                           location="query", required=True, default=3.4)
single_parser.add_argument("petal_l", type=float, help="Specify the petal lenght.",
                           location="query", required=True, default=1.6)
single_parser.add_argument("petal_w", type=float, help="Specify the sepal width.",
                           location="query", required=True, default=0.2)


# Define the single object classifier API
@api.route("/classify_iris/single", methods=["GET", "POST"])
class SingleClassifier(Resource):
    def predict(self):
        try:
            data = []
            data.append(float(request.args.get("sepal_l")))
            data.append(float(request.args.get("sepal_w")))
            data.append(float(request.args.get("petal_l")))
            data.append(float(request.args.get("petal_w")))
            return jsonify({"iris": predict_iris([data],
                            os.path.dirname(os.path.abspath(__file__)) + "/../model/model.pkl")})
        except KeyError as e:
            api.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            api.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    @api.doc(responses={200: "Success", 400: "Invalid Argument", 500: "Mapping Key Error"})
    @api.expect(single_parser, validate=True)
    def get(self):
        """Returns the predicted class."""
        return self.predict()

    @api.doc(responses={200: "Success", 400: "Invalid Argument", 500: "Mapping Key Error"})
    @api.expect(single_parser, validate=True)
    def post(self):
        """Returns the predicted class."""
        return self.predict()


# Define the multiple object classifier API
@api.route("/classify_iris/multiple", methods=["POST"])
class MultipleClassifier(Resource):
    def predict(self):
        try:
            json = request.get_json()["data"]
            return jsonify({"iris": predict_iris(json,
                            os.path.dirname(os.path.abspath(__file__)) + "/../model/model.pkl")})
        except KeyError as e:
            api.abort(500, e.__doc__, status="Could not retrieve information", statusCode="500")
        except Exception as e:
            api.abort(400, e.__doc__, status="Could not retrieve information", statusCode="400")

    @api.doc(responses={200: "Success", 400: "Invalid Argument", 500: "Mapping Key Error"})
    @api.expect(multiple_model, validate=True)
    def post(self):
        """Returns a list with the predicted classes."""
        return self.predict()


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 8000)))
