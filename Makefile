SHELL := /bin/bash

UV_VERSION     ?= 0.7.13
PYTHON_VERSION ?= 3.10.16
VENV           ?= radiologist
PYENV_GIT_TAG  ?= v2.6.3

PKG_CORE      := radiologist-core
PKG_ETL       := radiologist-etl
PKG_UTILS     := radiologist-utils
PKG_INFERENCE := radiologist-inference
PKG_REGISTRY  := radiologist-registry
PKG_CLI       := radiologist-cli

PYTEST_FLAGS ?= -q

define GPUENVVARS
# Hydra debug
export HYDRA_FULL_ERROR=1

# Set environment variables for GPU
export CUDA_HOME="/usr/local/cuda"
export CUDA_VERSION="12.5.1"
export CUDA_MAJOR_VERSION="12"
export CUDA_MINOR_VERSION="5"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64/stubs"
export LIBRARY_PATH="/usr/local/cuda/lib64/stubs"

export NVARCH="x86_64"

export NVIDIA_VISIBLE_DEVICES="all"
export NVIDIA_DRIVER_CAPABILITIES="compute,utility"

export NV_CUDA_CUDART_VERSION="12.5.82-1"
export NV_CUDA_CUDART_DEV_VERSION="12.5.82-1"
export NV_CUDA_LIB_VERSION="12.5.1-1"
export NV_CUDA_NSIGHT_COMPUTE_VERSION="12.5.1-1"
export NV_CUDA_NSIGHT_COMPUTE_DEV_PACKAGE="cuda-nsight-compute-12-5=12.5.1-1"
export NV_NVTX_VERSION="12.5.82-1"
export NV_NVPROF_VERSION="12.5.82-1"
export NV_NVPROF_DEV_PACKAGE="cuda-nvprof-12-5=12.5.82-1"

# cuDNN
export NV_CUDNN_VERSION="9.2.1.18-1"
export NV_CUDNN_PACKAGE="libcudnn9-cuda-12=9.2.1.18-1"
export NV_CUDNN_PACKAGE_DEV="libcudnn9-dev-cuda-12=9.2.1.18-1"

# cuBLAS
export NV_LIBCUBLAS_VERSION="12.5.3.2-1"
export NV_LIBCUBLAS_PACKAGE="libcublas-12-5=12.5.3.2-1"
export NV_LIBCUBLAS_PACKAGE_NAME="libcublas-12-5"
export NV_LIBCUBLAS_DEV_VERSION="12.5.3.2-1"
export NV_LIBCUBLAS_DEV_PACKAGE="libcublas-dev-12-5=12.5.3.2-1"
export NV_LIBCUBLAS_DEV_PACKAGE_NAME="libcublas-dev-12-5"

# NCCL (for multi-GPU)
export NCCL_VERSION="2.22.3-1"
export NV_LIBNCCL_PACKAGE="libnccl2=2.22.3-1+cuda12.5"
export NV_LIBNCCL_PACKAGE_NAME="libnccl2"
export NV_LIBNCCL_PACKAGE_VERSION="2.22.3-1"
export NV_LIBNCCL_DEV_PACKAGE="libnccl-dev=2.22.3-1+cuda12.5"
export NV_LIBNCCL_DEV_PACKAGE_NAME="libnccl-dev"
export NV_LIBNCCL_DEV_PACKAGE_VERSION="2.22.3-1"
endef


define TPUENVVARS
# Unset environment variables that are not needed
for var in MASTER_ADDR MASTER_PORT TPU_PROCESS_ADDRESSES XRT_TPU_CONFIG; do
    unset $var
done

# Hydra debug
export HYDRA_FULL_ERROR=1

# Set environment variables for TPU
export ISTPUVM=1
export PJRT_DEVICE=TPU
export PT_XLA_DEBUG_LEVEL=1
export TF_CPP_MIN_LOG_LEVEL=2
export TPU_ACCELERATOR_TYPE=v5litepod-8
export TPU_CHIPS_PER_HOST_BOUNDS=2,4,1
export TPU_HOST_BOUNDS=1,1,1
export TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434,8435,8436,8437,8438
export TPU_SKIP_MDS_QUERY=1
export TPU_WORKER_HOSTNAMES=localhost
export TPU_WORKER_ID=0
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000
endef


define PYENVINIT
# Pyenv setup

export PYENV_ROOT="$$HOME/.pyenv"
[[ -d $$PYENV_ROOT/bin ]] && export PATH="$$PYENV_ROOT/bin:$$PATH"
eval "$$(pyenv init - bash)"
eval "$$(pyenv virtualenv-init -)"
endef

define UVALIASES
# uv aliases

alias uvadd="uv add --active"
alias uvsync="uv sync --active"
endef

export GPUENVVARS
export TPUENVVARS
export PYENVINIT
export PYENV_GIT_TAG
export UVALIASES

.DEFAULT_GOAL := help


define confirm
	printf "$(1) [y/N] "; \
	read answer; \
	case "$$answer" in \
		y|Y|yes|YES) ;; \
		*) echo "Skipped."; exit 0 ;; \
	esac
endef

.PHONY: help \
        tpusetup gpusetup cpusetup \
        uv pyenv venv remove-tf tpuenvs gpuenvs reload \
        sync sync-all dev-install \
        build build-all \
        test test-core test-etl test-utils test-inference test-registry test-cli \
        lint format type-check \
        docs-install docs-serve docs-build docstrings \
        clean set-gpg-relay devcontainer-post-create \
		devcontainer-post-start verify-forwarding \
		debug-devcontainer summarize-egress debug-gpg-env

# --------------------------------------------------------------------------- #
#  Help                                                                        #
# --------------------------------------------------------------------------- #

debug-gpg-env:
	@echo "USER=$$USER"
	@echo "HOME=$$HOME"
	@echo "GNUPGHOME=$${GNUPGHOME:-<unset>}"
	@echo "uid=$$(id -u) gid=$$(id -g)"
	@echo "whoami=$$(whoami)"
	@gpgconf --list-dirs homedir
	@gpgconf --list-dirs socketdir
	@gpgconf --list-dirs agent-socket

set-gpg-relay:
	@$(call confirm, Create relay with docker ? will establish a TCP connexion for devcontainer to reuse your local key.)
	@.scripts/host-gpg-relay-install.sh

devcontainer-post-create:
	@.scripts/post-create.sh
	@echo "✅ post-create done."

devcontainer-post-start:
	@.scripts/post-start.sh
	@echo "✅ post-start done."

verify-forwarding:
	@.scripts/verify-forwarding.sh

debug-devcontainer: verify-forwarding
	@.scripts/gpg-reimport-stub.sh

set-remote-gpg-relay:
	@.scripts/gpg-relay.sh

summarize-egress:
	@.scripts/report-egress.sh to summarize
	@"✅ logs exported."

# --------------------------------------------------------------------------- #
#  Help                                                                        #
# --------------------------------------------------------------------------- #

help:  ## show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
#  Environment bootstrap (new machine / remote VM)                             #
# --------------------------------------------------------------------------- #

tpusetup: tpuenvs remove-tf uv pyenv venv reload  ## full TPU VM bootstrap
gpusetup: gpuenvs remove-tf uv pyenv venv reload  ## full GPU machine bootstrap
cpusetup: uv pyenv venv reload                    ## CPU-only bootstrap

uv:  ## install / upgrade uv to $(UV_VERSION)
	@echo "Installing uv $(UV_VERSION)..."
	@grep -q '# uv setup' ~/.bashrc || echo '# uv setup' >> ~/.bashrc
	@curl -LsSf https://astral.sh/uv/$(UV_VERSION)/install.sh | sh
	@grep -q 'uv generate-shell-completion bash' ~/.bashrc || echo 'eval "$$(uv generate-shell-completion bash)"' >> ~/.bashrc
	@grep -q 'uvx --generate-shell-completion bash' ~/.bashrc || echo 'eval "$$(uvx --generate-shell-completion bash)"' >> ~/.bashrc
	@grep -q '# uv aliases' ~/.bashrc || echo "$$UVALIASES" >> ~/.bashrc
	@echo "✅ uv installed!"

pyenv:  ## install pyenv
	@echo "Installing pyenv..."
	@curl https://pyenv.run | bash
	@grep -q 'pyenv init' ~/.bashrc || echo "$$PYENVINIT" >> ~/.bashrc
	@echo "✅ pyenv installed!"

venv:  ## create Python $(PYTHON_VERSION) virtualenv '$(VENV)' via pyenv
	@echo "Creating virtualenv '$(VENV)' (Python $(PYTHON_VERSION))..."
	@export PYENV_ROOT="$$HOME/.pyenv" && export PATH="$$PYENV_ROOT/bin:$$PATH" && eval "$$(pyenv init --path)" && eval "$$(pyenv init -)" && \
	if ! pyenv versions --bare | grep -q "^$(PYTHON_VERSION)$$"; \
	then pyenv install $(PYTHON_VERSION); \
	else echo "✅ Python $(PYTHON_VERSION) already installed"; fi
	@export PYENV_ROOT="$$HOME/.pyenv" && export PATH="$$PYENV_ROOT/bin:$$PATH" && eval "$$(pyenv init --path)" && eval "$$(pyenv init -)" && \
	if ! pyenv virtualenvs --bare | grep -q "^$(VENV)$$"; \
	then pyenv virtualenv $(PYTHON_VERSION) $(VENV); \
	else echo "✅ Virtualenv '$(VENV)' already exists"; fi
	@echo "✅ Virtualenv ready — run: pyenv activate $(VENV)"

remove-tf:  ## uninstall tensorflow family to avoid conflicts
	@echo "Removing tensorflow packages..."
	@uv pip uninstall tensorflow tensorflow-tpu tensorboard -y 2>/dev/null || true
	@echo "✅ tensorflow packages removed"

tpuenvs:  ## append TPU env vars to ~/.bashrc
	@echo "Setting up TPU environment variables..."
	@grep -q '# Set environment variables for TPU' ~/.bashrc || echo "$$TPUENVVARS" >> ~/.bashrc
	@echo "✅ TPU env vars written"

gpuenvs:  ## append GPU / CUDA env vars to ~/.bashrc
	@echo "Setting up GPU environment variables..."
	@grep -q '# Set environment variables for GPU' ~/.bashrc || echo "$$GPUENVVARS" >> ~/.bashrc
	@echo "✅ GPU env vars written"

reload:  ## remind to reload shell
	@echo "⏭️  Run: source ~/.bashrc"

# --------------------------------------------------------------------------- #
#  Day-to-day dev                                                              #
# --------------------------------------------------------------------------- #

sync:  ## sync all dep-groups + all workspace packages (no optional extras)
	@uv sync --active --all-groups --all-packages

sync-all:  ## sync all dep-groups + all workspace packages + all optional extras except ray (deferred backend, see #188)
	@uv sync --active --all-groups --all-packages --all-extras --no-extra ray

dev-install: sync-all  ## sync deps + install pre-commit hooks (run once after clone)
	@uv run --active pre-commit install
	@uv run --active pre-commit install --install-hooks -t commit-msg
	@echo "✅ Dev environment ready!"

# --------------------------------------------------------------------------- #
#  Build                                                                       #
# --------------------------------------------------------------------------- #

build:  ## build a single distribution — usage: make build PKG=radiologist-core
	@test -n "$(PKG)" || (echo "PKG is required, e.g. make build PKG=radiologist-core" && exit 1)
	@uv build --package $(PKG) --out-dir dist

build-all:  ## build all seven distributions
	@uv build --package radiologist --out-dir dist
	@uv build --package radiologist-core --out-dir dist
	@uv build --package radiologist-etl --out-dir dist
	@uv build --package radiologist-inference --out-dir dist
	@uv build --package radiologist-registry --out-dir dist
	@uv build --package radiologist-utils --out-dir dist
	@uv build --package radiologist-cli --out-dir dist

# --------------------------------------------------------------------------- #
#  Tests                                                                       #
# --------------------------------------------------------------------------- #

test:  ## run all package tests
	@uv run --active pytest $(PYTEST_FLAGS)

test-core:  ## run radiologist-core tests only
	@uv run --active pytest $(PKG_CORE)/radiologist_core_tests $(PYTEST_FLAGS)

test-etl:  ## run radiologist-etl tests only
	@uv run --active pytest $(PKG_ETL)/radiologist_etl_tests $(PYTEST_FLAGS)

test-utils:  ## run radiologist-utils tests only
	@uv run --active pytest $(PKG_UTILS)/radiologist_utils_tests $(PYTEST_FLAGS)

test-inference:  ## run radiologist-inference tests only
	@uv run --active pytest $(PKG_INFERENCE)/radiologist_inference_tests $(PYTEST_FLAGS)

test-registry:  ## run radiologist-registry tests only
	@uv run --active pytest $(PKG_REGISTRY)/radiologist_registry_tests $(PYTEST_FLAGS)

test-cli:  ## run radiologist-cli tests only
	@uv run --active pytest $(PKG_CLI)/radiologist_cli_tests $(PYTEST_FLAGS)

# --------------------------------------------------------------------------- #
#  Code quality                                                                #
# --------------------------------------------------------------------------- #

lint:  ## run all pre-commit hooks on every file
	@uv run --active pre-commit run --all-files

format:  ## run black + isort in-place
	@uv run --active black .
	@uv run --active isort .

type-check:  ## run mypy across all packages
	@uv run --active mypy $(PKG_CORE)/src $(PKG_ETL)/src $(PKG_UTILS)/src $(PKG_INFERENCE)/src $(PKG_REGISTRY)/src $(PKG_CLI)/src

# --------------------------------------------------------------------------- #
#  Documentation                                                               #
# --------------------------------------------------------------------------- #

docs-install:  ## sync docs deps + all extras (mkdocstrings must import CLI modules)
	@uv sync --active --group docs --all-packages --all-extras

docs-serve:  ## live-reload docs at localhost:8000
	@uv run --active mkdocs serve

docs-build:  ## strict build — fails on broken refs / missing docstrings pages
	@uv run --active mkdocs build --strict

docstrings:  ## Google-style docstring check (also enforced via pre-commit's flake8 hook)
	@uv run --active flake8 --select=D radiologist-*/src

# --------------------------------------------------------------------------- #
#  Maintenance                                                                 #
# --------------------------------------------------------------------------- #

clean:  ## remove __pycache__, .pytest_cache, .mypy_cache, .coverage, dist
	@find . -type d \( -name '__pycache__' -o -name '.pytest_cache' -o -name '.mypy_cache' \) \
		-not -path './.git/*' -exec rm -rf {} + 2>/dev/null || true
	@rm -f .coverage coverage.xml
	@rm -rf dist
	@echo "✅ Clean!"
