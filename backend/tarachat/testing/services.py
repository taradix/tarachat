"""Service fixtures for docker compose integration tests."""

from functools import partial
from pathlib import Path

import pytest
from yarl import URL

from tarachat.testing.compose import ComposeServer


@pytest.fixture(scope="session")
def project():
    return "test"


@pytest.fixture(scope="session")
def env_file(project, request):
    """Environment file for compose services."""
    env_file = request.config.cache.makedir("compose") / "env"
    with env_file.open("w") as f:
        f.write(f"COMPOSE_PROJECT_NAME={project}\n")
    return env_file


@pytest.fixture(scope="session")
def compose_files(request):
    directory = Path(request.config.rootdir)
    filenames = ["docker-compose.yml", "compose.yaml", "compose.yml"]
    while True:
        for filename in filenames:
            path = directory / filename
            if path.exists():
                all_files = directory.glob(f"{path.stem}.*")
                ordered_files = sorted(all_files, key=lambda p: len(p.name))
                return list(ordered_files)

        if directory == directory.parent:
            raise FileNotFoundError("Docker compose file not found")

        directory = directory.parent


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
    from tarachat.testing.http import HTTPSession

    url = URL.build(scheme="http", host=backend_service.ip, port=8000)
    return HTTPSession(url)
