VENV := .venv

# Prevent uv from re-syncing, which would revert CUDA torch to CPU.
RUN := uv run --no-sync
PYTHON := $(RUN) python
TOUCH := $(PYTHON) -c 'import sys; from pathlib import Path; Path(sys.argv[1]).touch()'
HAS_NVIDIA := $(shell nvidia-smi > /dev/null 2>&1 && echo 1 || echo 0)

uv.lock: pyproject.toml
	uv lock

# Build .venv with deps.
$(VENV): uv.lock
	@echo "==> Installing environment..."
ifeq ($(HAS_NVIDIA),1)
	@echo "==> NVIDIA GPU detected, installing CUDA-enabled PyTorch..."
	@uv sync --frozen --all-extras --no-install-package torch
	@uv pip install torch --index-url https://download.pytorch.org/whl/cu128
else
	@echo "==> No NVIDIA GPU detected, using CPU-only PyTorch."
	@uv sync --frozen --all-extras
endif
	@$(TOUCH) $@

# Convenience target to build venv
.PHONY: setup
setup: $(VENV)

.PHONY: check
check: $(VENV)
	@echo "==> Checking uv lock..."
	@uv lock --check
	@echo "==> Linting Python code..."
	@$(RUN) ruff check .

.PHONY: test
test: $(VENV)
	@echo "==> Testing Python code..."
	@$(RUN) coverage erase
	@$(RUN) coverage run -p -m pytest

.PHONY: coverage
coverage: $(VENV)
	@echo "==> Checking coverage..."
	@$(RUN) coverage combine
	@$(RUN) coverage html --skip-covered --skip-empty
	@$(RUN) coverage report

.DEFAULT_GOAL := test
