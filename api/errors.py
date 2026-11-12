import uuid
from flask import jsonify, g

class ApiError(Exception):
    def __init__(self, code: str, status_code: int, message: str, details: dict = None):
        super().__init__(message)
        self.code = code
        self.status_code = status_code
        self.message = message
        self.details = details or {}

def register_error_handlers(app):
    @app.errorhandler(ApiError)
    def handle_api_error(error):
        response = {
            "error": {
                "code": error.code,
                "message": error.message,
                "details": error.details
            }
        }
        return jsonify(response), error.status_code

    @app.errorhandler(404)
    def handle_404(e):
        return jsonify({
            "error": {
                "code": "not_found",
                "message": "Resource not found",
                "details": {}
            }
        }), 404

    @app.errorhandler(Exception)
    def handle_unexpected(e):
        req_id = getattr(g, "request_id", uuid.uuid4().hex)
        app.logger.exception(f"Unhandled exception [req_id={req_id}]: {str(e)}")
        return jsonify({
            "error": {
                "code": "internal_error",
                "message": "An unexpected error occurred",
                "details": {"request_id": req_id}
            }
        }), 500
