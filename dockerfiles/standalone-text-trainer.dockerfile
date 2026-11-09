FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# The base already includes CUDA/compiler tools. Interactive editors/monitors
# are not training dependencies; avoid a runtime image build relying on apt mirrors.
ENV PIP_NO_CACHE_DIR=1 UV_NO_CACHE=1
# Default dir
RUN mkdir -p /workspace
RUN mkdir -p /cache
RUN mkdir -p /workspace/scripts/datasets
RUN mkdir -p /app/checkpoints
WORKDIR /workspace/scripts
COPY scripts/training_requirements.txt /workspace/scripts/training_requirements.txt
# Pytorch (Auto-selects backend https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)

# Create a virtual environment for data generation
RUN python -m venv /workspace/axo_py
RUN bash -c "source /workspace/axo_py/bin/activate && \
    pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install --no-build-isolation axolotl==0.9.1 && \
    deactivate"


# install the main dependencies
RUN pip install uv && \
    pip install -U packaging==23.2 setuptools==75.8.0 wheel ninja && \
    uv pip install -r /workspace/scripts/training_requirements.txt --system && \
    pip install hf_transfer==0.1.9 && \
    pip install tenacity==9.1.2 && \
    pip install tiktoken==0.9.0 && \
    uv pip install vllm==0.8.3 --system && \
    pip install flash-attn==v2.7.4.post1 --no-build-isolation && \
    pip install numpy==1.26.3 pyarrow==19.0.1

# Build the architecture-specific dependency overlay while internet is available.
# Never uninstall or download packages inside the restricted training container.
RUN python -m venv --system-site-packages /workspace/oss_py && \
    /workspace/oss_py/bin/pip install transformers==4.55.0 deepspeed==0.17.4 kernels==0.9.0
RUN python -m pip check && /workspace/axo_py/bin/python -m pip check && /workspace/oss_py/bin/python -m pip check
RUN python -m pip freeze > /workspace/main-dependencies.txt && \
    /workspace/oss_py/bin/pip freeze > /workspace/oss-dependencies.txt
RUN python -c "import numpy, pyarrow, datasets; assert numpy.__version__ == '1.26.3'; assert pyarrow.__version__ == '19.0.1'"
COPY scripts /workspace/scripts
RUN chmod +x /workspace/scripts/entrypoint.sh
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline
ENV HF_HOME=/app/checkpoints/.cache/huggingface \
    XDG_CACHE_HOME=/app/checkpoints/.cache \
    TRITON_CACHE_DIR=/app/checkpoints/.cache/triton \
    TORCHINDUCTOR_CACHE_DIR=/app/checkpoints/.cache/inductor
ENTRYPOINT ["./entrypoint.sh"]
