"""Service fixtures for docker compose integration tests."""

from functools import partial

import httpx
import pytest
from pytest_xdocker.compose import ComposeServer, find_compose_files
from yarl import URL

from tarachat.testing.http import HTTPSession


@pytest.fixture(scope="session")
def env_file(request):
    """Environment file for compose services."""
    env_file = request.config.cache.makedir("tarachat") / "env"
    with env_file.open("w") as f:
        f.write("SERVER_HOSTNAME=test.local\n")
    return env_file


@pytest.fixture(scope="session")
def compose_server(env_file, process, request):
    return partial(
        ComposeServer,
        env_file=env_file,
        compose_files=find_compose_files(request.config.rootdir),
        process=process,
    )


@pytest.fixture(scope="session")
def backend_service(compose_server):
    """Backend service fixture."""
    server = compose_server("Application startup complete")
    with server.run("backend") as service:
        yield service


@pytest.fixture(scope="session")
def api_session(backend_service):
    """HTTP session connected to the backend API."""
    url = URL.build(scheme="http", host=backend_service.ip, port=8000)
    return HTTPSession(url)


class NoDepsComposeServer(ComposeServer):
    """Compose server that doesn't start the service dependencies."""

    def prepare_func(self, controldir):
        data = super().prepare_func(controldir)
        return data._replace(args=data.args.with_optionals("--no-deps"))


@pytest.fixture(scope="session")
def certbot_service(compose_server):
    """Certbot service fixture, with a self-signed dev cert."""
    server = compose_server("Ready")
    with server.run("certbot") as service:
        yield service


@pytest.fixture(scope="session")
def nginx_service(certbot_service, compose_server):
    """Nginx service fixture.

    Skips the backend and ollama dependencies, which need an nvidia runtime.
    """
    server = NoDepsComposeServer("start worker process", **compose_server.keywords)
    with server.run("nginx") as service:
        yield service


@pytest.fixture
def nginx_client(nginx_service):
    """Client to the nginx service fixture, over the self-signed dev cert."""
    base_url = f"https://{nginx_service.ip}"
    with httpx.Client(base_url=base_url, timeout=5.0, verify=False) as client:  # noqa: S501
        yield client
