# 踩坑笔记（搭建与联调实录）

> 项目：企业级多智能体 AI 助手系统（马小i 参考架构）
> 记录搭建/联调过程中实际踩到的坑，每条含【现象 → 根因 → 解决】。
> 相关修复已固化在代码中，重启/重建环境时按此文档排查可少走弯路。

---

## 一、Docker 构建与网络

### 1. Docker Hub 拉不到基础镜像（auth.docker.io 超时）

**现象**

```
target hr-agent: failed to solve: failed to fetch oauth token: Post "https://auth.docker.io/token":
dial tcp [2a03:...]:443: connectex: A connection attempt failed...
```

**根因**：`python:3.11-slim` 解析到 Docker Hub 官方源，当前网络无法访问 `auth.docker.io`（IPv6 连接超时）。

**解决**：
- [docker/Dockerfile](docker/Dockerfile) 基础镜像改为参数化，默认走 DaoCloud 镜像源：
  `ARG BASE_IMAGE=docker.m.daocloud.io/library/python:3.11-slim`
- pip 同步切清华源：`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`
- [docker/docker-compose.yml](docker/docker-compose.yml) 通过 `x-build` 锚点统一注入两个 build-args，可用环境变量覆盖
- 备选镜像源：`docker.1ms.run/python:3.11-slim`、`mirror.baidubce.com/library/python:3.11-slim`

### 2. 容器内 `localhost` 不是宿主机（OLLAMA_BASE_URL 陷阱）

**现象**：容器内访问 Ollama 连接拒绝/超时（ingest 第一步 embedding 就失败）。

**根因**：[docker/.env](docker/.env) 里最初写的是 `OLLAMA_BASE_URL=http://localhost:11434`，
容器内的 `localhost` 指向**容器自身**，不是宿主机。

**解决**：改为 `http://host.docker.internal:11434`；compose 中各服务已配置
`extra_hosts: ["host.docker.internal:host-gateway"]`（Linux 容器必需）。

**教训**：凡是"容器访问宿主机服务"的配置，一律用 `host.docker.internal`，review .env 时重点盯 `localhost`。

---

## 二、依赖大版本陷阱（本次连环坑主因）

> 教训：requirements 只写 `pkg>=x` 无上界，构建时会滚动到最新大版本，
> API 不兼容直接启动崩溃。**生产环境必须锁大版本上界。**

### 3. a2a-sdk 1.x 改用 protobuf 底座，API 全变

**现象**（assistant / hr-agent / finance-agent 启动即崩）：

```
ImportError: cannot import name 'A2AClient' from 'a2a.client'
ModuleNotFoundError: No module named 'a2a.server.apps'
```

**根因**：PyPI 已发 a2a-sdk 1.1.2，1.x 基于 protobuf（`a2a_pb2`）重写，
`A2AClient`/`A2AStarletteApplication`/`TextPart`/`MessageSendParams` 等 0.2/0.3 API 全部移除。

**解决**：[requirements.txt](requirements.txt) 锁定
`a2a-sdk[http-server]>=0.3.0,<0.4.0`（实测 0.3.26）。
选 0.3.x 还因为蓝图要求 **JSON-RPC 2.0 over Streamable HTTP + SSE**，正是 0.3.x 的传输形态。

**验证方法**（临时容器，不污染本地）：

```bash
docker run --rm -v "$PWD/tmp_verify.py:/tmp/v.py" <base-image> \
  sh -c "pip install -q -i <mirror> 'a2a-sdk[http-server]>=0.3,<0.4' && python /tmp/v.py"
```

### 4. mcp 2.x 把 FastMCP 改名为 MCPServer

**现象**：

```
ModuleNotFoundError: No module named 'mcp.server.fastmcp'.
This is mcp 2.x, where FastMCP was renamed to MCPServer ... pin 'mcp<2' to keep running v1 code.
```

**根因**：MCP 官方 Python SDK 2.x 重构，`mcp.server.fastmcp.FastMCP` → `mcp.server.mcpserver.MCPServer`。

**解决**：锁定 `mcp>=1.10.0,<2`。蓝图明确要求"Server 端用内置 FastMCP 高层 API"，1.x 才有。

### 5. pymilvus 3.x 与环境变量 `MILVUS_URI` 撞车

**现象**（assistant 启动即崩）：

```
pymilvus.exceptions.ConnectionConfigException: Illegal uri: [/data/milvus_lite.db],
expected form 'http[s]://[user:password@]example.com[:12345]'
```

**根因**：compose 里我们自己设了环境变量 `MILVUS_URI=/data/milvus_lite.db`，
而 pymilvus **自身也会读取同名环境变量**，且在 import 时就校验必须是 http(s) 地址 → 导入即崩。
注意报错发生在 import 阶段（`pymilvus.orm.connections` 模块级初始化），还没走到我们的代码。

**解决**：
- 我们的环境变量改名 `MILVUS_LITE_URI`（[config.py](app/config.py)、[docker-compose.yml](docker/docker-compose.yml)、[.env](.env)）
- 锁定 `pymilvus[milvus_lite]>=2.4.8,<3`

**教训**：自定义环境变量避开知名三方库的保留变量名（`MILVUS_URI`/`HF_HOME`/`HTTP_PROXY` 等）。

---

## 三、Milvus Lite

### 6. milvus-lite 成为可选依赖（pymilvus 2.5+）

**现象**（ingest 报错）：

```
ModuleNotFoundError: No module named 'milvus_lite'
pymilvus.exceptions.ConnectionConfigException: milvus-lite is required for local database
connections. Please install it with: pip install pymilvus[milvus_lite]
```

**根因**：pymilvus 2.5 起，Milvus Lite 从默认依赖拆分为 extras。

**解决**：`pymilvus[milvus_lite]>=2.4.8,<3`（extras 写法，容器内自动装上 milvus-lite）。

### 7. 集合 `released` 状态不跨进程保留

**现象**（Web 问答报错）：

```
MilvusException: (code=101, message=Collection 'enterprise_knowledge' is in state 'released';
call load() before search/get/query)
```

**根因**：Milvus Lite 集合的 loaded 状态不落盘。ingest 进程里 load 过，进程退出后，
assistant 进程重新打开同一 db 文件时集合回到 released，search/query 直接抛错。

**解决**：[vectorstore.py](app/rag/vectorstore.py) `_ensure_collection()` 里**每次打开 db 都显式
`load_collection()`**（已存在则直接 load，新建后也 load）。

**教训**：嵌入式向量库的"内存态"与"文件态"不一致，任何新进程打开都要先 load。

### 8. Milvus Lite 的路径与平台限制

- **db 文件无需预先存在**：`MilvusClient(uri=...)` 首次使用自动创建，但**父目录必须存在**；
- **相对路径按进程 CWD 解析**：不在项目根启动就会建错位置。
  [vectorstore.py](app/rag/vectorstore.py) 的 `_resolve_uri()` 统一锚定项目根 + 自动 mkdir；
- **不支持 Windows 宿主机**：本机直跑 `uvicorn` 走不通（pymilvus 导入 milvus-lite 即失败），
  必须走 Docker（Linux 容器）；生产换 Milvus Server 只需改 `MILVUS_LITE_URI` 为 `http://...`。

---

## 四、Ollama

### 9. Windows 版 Ollama 跑 bge-reranker-v2-m3 直接崩溃

**现象**（KB 问答时 /api/embed 返回 500）：

```
{"error":"llama-server process has terminated: exit status 0xc0000409:
The system detected an overrun of a stack-based buffer..."}
```

**根因**：Windows 版 Ollama 内置 llama.cpp 与该 reranker GGUF（XLM-RoBERTa 架构）存在兼容性 bug，
`/api/embed` 与 `/api/embeddings` 两个端点均崩，属 Ollama 侧问题，应用层无法绕过。

**解决**（应用层优雅降级）：
- [retriever.py](app/rag/retriever.py) rerank 包 try/except，失败自动回退 **RRF 融合排序**（混合检索结果依然可用），并打 warning 日志；
- 新增开关 `RERANK_ENABLED`（[config.py](app/config.py)），确认环境跑不动时设 `false` 跳过，省去每次约 12s 的崩溃等待；
- 治本选项：升级宿主机 Ollama 重试 / reranker 换 TEI、vLLM 等原生 rerank 服务（接口隔离在 [reranker.py](app/rag/reranker.py)，可无缝替换）。

**附带发现**：模型首次冷加载很慢（reranker 首请求 5s 内必超时），客户端超时要给足（项目内统一 120s+）。

---

## 五、Windows 开发环境杂项

### 10. PowerShell 转义：内嵌 JSON / 多行 python -c 必被 mangle

**现象**：`curl -d "{\"a\":1}"` 报 JSON decode error；docker exec 传多行 `python -c "..."` 报
`The 'from' keyword is not supported in this version of the language`（PowerShell 把引号内容按自身语法拆了）。

**解决**：
- JSON 请求体写入文件后 `curl --data-binary "@file"`；
- 容器内验证脚本先落盘，再 `-v 挂载: /tmp/v.py` 执行，绝不内联多行 python。

### 11. compose 构建竞态

**现象**：一次 `up -d --build` 尚未结束时又发起第二次，两次构建/重建容器相互干扰。

**解决**：串行执行；等待方式用"日志文件 mtime 超过 N 秒未更新"判断构建结束（TaskOutput 查不到
run_in_background 任务 id 时，直接 tail 输出文件）。

---

## 经验总结

1. **依赖锁大版本上界**：`a2a-sdk<0.4`、`mcp<2`、`pymilvus<3` 都是血泪换来的；
   新技术栈（MCP/A2A）版本演进极快，锁版本 + 注释原因。
2. **验证 API 兼容性用一次性容器**：`docker run --rm + pip install + python 脚本`，
   不污染本地环境，几分钟内确认目标版本的导入面。
3. **报错先看import栈**：`site-packages` 里的栈帧说明是依赖版本问题，不是业务代码问题。
4. **镜像源/索引全部参数化**：`BASE_IMAGE`、`PIP_INDEX_URL` 走 build-arg + .env，换源不改文件。
5. **容器访问宿主机一律 `host.docker.internal`**，并确认 compose 有 `host-gateway` 映射。
6. **外部依赖（Ollama/向量库）调用都要有降级路径**：rerank 崩了回退 RRF，系统核心链路不中断。
