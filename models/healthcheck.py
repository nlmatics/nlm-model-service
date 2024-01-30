from flask_restful import Resource


class HealthCheck(Resource):
    def get(self):
        return "Healthy!", 200

    def post(self):
        return "Healthy!", 200
