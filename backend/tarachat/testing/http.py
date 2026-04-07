"""HTTP session for integration tests."""

import httpx
from yarl import URL


class HTTPSession:
    """HTTP session for making requests to a service."""

    def __init__(self, base_url: URL):
        self.base_url = base_url
        self.client = httpx.Client(base_url=str(base_url), timeout=30)

    def get(self, path: str, **kwargs):
        response = self.client.get(path, **kwargs)
        response.raise_for_status()
        return response

    def post(self, path: str, **kwargs):
        response = self.client.post(path, **kwargs)
        response.raise_for_status()
        return response

    def close(self):
        self.client.close()
