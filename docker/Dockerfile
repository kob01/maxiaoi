ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim

FROM ${BASE_IMAGE}

WORKDIR /srv

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/srv \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    # 强制使用镜像自带的系统 Python, 禁止 uv 联网下载自有 Python
    UV_PYTHON_PREFERENCE=only-system \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/srv/.venv
ENV PATH="/srv/.venv/bin:$PATH"

# PyPI 镜像源: pip 安装 uv 与 uv 下载依赖共用
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    UV_INDEX_URL=${PIP_INDEX_URL}

RUN pip install --no-cache-dir uv

# 依赖层: 仅当 pyproject.toml / uv.lock 变化时才重建
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

COPY app ./app
COPY web ./web
COPY scripts ./scripts
COPY data/knowledge ./data/knowledge

EXPOSE 8000 8001 8002 9001 9002

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
