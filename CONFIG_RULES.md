# 配置红线记录（每次改动前必读）

## 1. `docker/.env` — `OLLAMA_BASE_URL`（禁止修改）

```ini
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

- **规则**：此行保持原样，任何改动（代码重构、compose 调整、环境变量整理）都不得把它改回 `localhost` 或删除。
<!-- - **原因**：assistant 及各 MCP/A2A 服务运行在 Docker 容器内，容器里的 `localhost` 指向**容器自身**而不是宿主机；访问宿主机 Ollama 必须走 `host.docker.internal`。
- **事故记录（2026-09-17）**：该值一度配成 `http://localhost:11434`，容器内无法连到 Ollama，上传图片立即 502，报错 `All connection attempts failed`（0.1s 内失败，非超时）。改回 `host.docker.internal` 并重建容器后恢复。 -->
- **检查时机**：凡涉及 `docker/.env`、`app/config.py`、`docker/docker-compose.yml` environment 段的改动，或排查任何 Ollama 连接类报错时，先核对本条。

