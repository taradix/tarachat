"""Service fixtures for docker compose integration tests."""

from functools import partial

import pytest
from pytest_xdocker.compose import ComposeServer, find_compose_files
from yarl import URL

from tarachat.testing.http import HTTPSession


@pytest.fixture(scope="session")
def project():
    return "test"


@pytest.fixture(scope="session")
def env_file(project, request):
    """Environment file for compose services."""
    env_file = request.config.cache.makedir("tarachat") / "env"
    with env_file.open("w") as f:
        f.write(f"COMPOSE_PROJECT_NAME={project}\n")
    return env_file


@pytest.fixture(scope="session")
def compose_files(request):
    return find_compose_files(request.config.rootdir)


@pytest.fixture(scope="session")
def compose_server(project, env_file, compose_files, process):
    return partial(
        ComposeServer,
        project=project,
        env_file=env_file,
        compose_files=compose_files,
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
