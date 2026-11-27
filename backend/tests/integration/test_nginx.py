"""Integration tests for the nginx service."""

import httpx


def test_nginx_redirects_http_to_https(nginx_service):
    """The nginx service should redirect HTTP to HTTPS."""
    response = httpx.get(f"http://{nginx_service.ip}/api/health")

    assert response.status_code == 301
    assert response.headers["location"] == f"https://{nginx_service.ip}/api/health"


def test_nginx_status(nginx_client):
    """The nginx service should expose its status to private networks."""
    response = nginx_client.get("/nginx_status")

    assert response.status_code == 200
    assert "Active connections" in response.text


def test_nginx_proxies_api(nginx_client):
    """The nginx service should proxy /api/ to the backend, which isn't running."""
    response = nginx_client.get("/api/health")

    assert response.status_code == 502
