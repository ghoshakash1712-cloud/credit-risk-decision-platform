import os
import time
import uuid
import json
import logging
from flask import Flask, g, request
from api.loaders import Store
from api.errors import register_error_handlers
from api.routes.v1 import v1

STATIC_ENDPOINTS = {
    "/v1/meta",
    "/v1/model-report",
    "/v1/explain/global",
    "/v1/drift"
}

def create_app(cfg=None, store=None):
    app = Flask(__name__)
    
    # Configure logging
    log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s")
    logger = logging.getLogger("crs_api")
    app.logger = logger
    
    # Initialize store if not provided
    if store is None:
        logger.info("Initializing Store and running self-check...")
        store = Store(cfg)
        logger.info(
            f"Store initialized successfully: model_version={store.meta.get('model_version')}, "
            f"data_release={store.meta.get('data_release')}, holdout_loans={len(store.holdout)}"
        )
    app.extensions["store"] = store
    
    # Register blueprints and handlers
    app.register_blueprint(v1, url_prefix="/v1")
    register_error_handlers(app)
    
    @app.before_request
    def before_request():
        g.request_id = uuid.uuid4().hex
        g.start = time.perf_counter()
        
    @app.after_request
    def after_request(resp):
        req_id = getattr(g, "request_id", uuid.uuid4().hex)
        resp.headers["X-Request-ID"] = req_id
        
        # AP-06: caching headers
        if request.path in STATIC_ENDPOINTS:
            resp.headers["Cache-Control"] = "max-age=60"
        else:
            resp.headers["Cache-Control"] = "no-store"
            
        # DT-06: structured log line
        duration_ms = (time.perf_counter() - getattr(g, "start", time.perf_counter())) * 1000.0
        log_entry = {
            "request_id": req_id,
            "method": request.method,
            "endpoint": request.path,
            "status": resp.status_code,
            "duration_ms": round(duration_ms, 2)
        }
        logger.info(json.dumps(log_entry))
        return resp
        
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=8000, debug=False)
