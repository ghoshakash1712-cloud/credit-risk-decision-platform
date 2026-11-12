import os
import urllib.parse
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

class ApiUnreachable(Exception):
    pass

class ApiRejected(Exception):
    def __init__(self, error):
        self.error = error
        super().__init__(error.get("message", "API Error"))

@st.cache_data(ttl=60, show_spinner=False)
def _fetch_cached(full_url: str, params_items: tuple):
    params = dict(params_items)
    try:
        resp = requests.get(full_url, params=params, timeout=5)
    except (requests.ConnectionError, requests.Timeout) as e:
        raise ApiUnreachable(str(e))
        
    if not resp.ok:
        try:
            err_data = resp.json().get("error", {"message": resp.text, "code": "http_error"})
        except Exception:
            err_data = {"message": f"HTTP {resp.status_code}: {resp.text}", "code": "http_error"}
        raise ApiRejected(err_data)
        
    return resp.json()

def get(path: str, **params):
    clean_params = {k: v for k, v in params.items() if v is not None}
    full_url = f"{API_URL.rstrip('/')}/v1{path}"
    items = tuple(sorted(clean_params.items()))
    return _fetch_cached(full_url, items)

def safe(fn, *args, **kwargs):
    cache_key = f"{fn.__name__}_{args}_{kwargs}"
    try:
        result = fn(*args, **kwargs)
        st.session_state[f"last_good_{cache_key}"] = result
        return result, False
    except ApiUnreachable:
        st.error("⚠️ Backend API is unreachable. Please verify that the API service is running on " + API_URL)
        if f"last_good_{cache_key}" in st.session_state:
            st.warning("⚠️ Showing previous cached data (stale).")
            return st.session_state[f"last_good_{cache_key}"], True
        return None, True
    except ApiRejected as e:
        st.error(f"⚠️ Request Rejected: {e.error.get('message', 'Invalid parameter')}")
        if f"last_good_{cache_key}" in st.session_state:
            st.warning("⚠️ Showing previous cached data (stale).")
            return st.session_state[f"last_good_{cache_key}"], True
        return None, True
