"""Compose server module."""

from contextlib import contextmanager
from datetime import datetime

from attrs import define
from more_itertools import only
from pytest_xdocker.docker import DockerContainer
from pytest_xdocker.process import ProcessData, ProcessServer
from pytest_xdocker.xdocker import xdocker


@define(frozen=True)
class ComposeService:
    """Compose service.

    :param name: Name of the compose service container.
    """

    name: str

    @property
    def container(self):
        return DockerContainer(self.name)

    @property
    def container_id(self):
        return self.container.inspect["Id"]

    @property
    def env(self):
        env = self.container.inspect["Config"]["Env"]
        return dict(e.split("=", 1) for e in env)

    @property
    def ip(self):
        network_settings = self.container.inspect["NetworkSettings"]
        network = only(network_settings["Networks"].values())
        return network["IPAddress"]

    @property
    def started_at(self):
        started_at = self.container.inspect["State"]["StartedAt"]
        return datetime.fromisoformat(started_at)


class ComposeServer(ProcessServer):
    def __init__(self, pattern, project="test", env_file=None, compose_files=None, timeout=180, **kwargs):
        """Initialize a compose server."""
        super().__init__(**kwargs)
        self.pattern = pattern
        self.project = project
        self.env_file = env_file
        self.compose_files = compose_files
        self.timeout = timeout

    def __repr__(self):
        return f"{self.__class__.__name__}(pattern={self.pattern!r}, project={self.project!r})"

    def full_name(self, name):
        return f"{self.project}-{name}-1"

    def prepare_func(self, controldir):
        """Prepare the function to run the compose service."""
        full_name = self.full_name(controldir.basename)
        compose = xdocker.compose().with_project_name(self.project)
        if env_file := self.env_file:
            compose = compose.with_env_file(env_file)
        for file in self.compose_files or []:
            compose = compose.with_file(file)

        command = (
            compose
            .run(controldir.basename)
            .with_name(full_name)
            .with_build()
            .with_remove()
            .with_optionals("--use-aliases")
        )

        return ProcessData(self.pattern, command, timeout=self.timeout)

    @contextmanager
    def run(self, name):
        """Return a `ComposeService` to the running service."""
        with super().run(name):
            full_name = self.full_name(name)
            yield ComposeService(full_name)
