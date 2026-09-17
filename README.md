# 马小i · 企业级多智能体 AI 助手系统

参考马上消费"马小i"公开技术架构,实现 **Assistant-Agent 一入口多智能体** 模式:
用户只面对唯一 Assistant,由其按任务复杂度分层调度 —— 知识库直答(RAG)、
MCP 工具调用、A2A 专业智能体委派。

## 架构

```
                ┌──────────────────────── Web / API ─────────────────────────┐
                │                      Assistant (统一入口)                   │
                │  FastAPI + LangGraph 编排: 意图识别(qwen3.5) → 分层路由   │
                └───┬───────────────┬───────────────────┬────────────────────┘
                    │               │                   │
            a. 简单查询      b. 复杂操作(MCP)      c. 专业任务(A2A)
                    │               │                   │
          ┌─────────▼────┐  ┌───────▼───────┐   ┌───────▼────────┐
          │  RAG 知识底座 │  │  MCP Servers  │   │ 专业 Agent      │
          │ bge-m3 向量  │  │ HR工单 :8001  │   │ HR_Agent :9001 │
          │ Milvus Lite  │  │ 财务报销:8002 │   │ Finance  :9002 │
          │ BM25+Rerank  │  │ (FastMCP)     │   │ (LangGraph+MCP)│
          └──────────────┘  └───────────────┘   └────────────────┘
                    └──── 底座: Ollama (qwen3.5 / bge-m3 / bge-reranker-v2-m3) ────┘
```

A2A 遵循 Agent2Agent 协议:Agent Card 发布于 `/.well-known/agent-card.json`,
通信为 JSON-RPC 2.0 over HTTP(`message/send`)。MCP 遵循 Model Context
Protocol:FastMCP server,streamable-http transport(`:8001/mcp`、`:8002/mcp`)。

## 目录结构

```
mxi/
├── app/
│   ├── config.py                 # 全局配置(pydantic-settings)
│   ├── schemas.py                # 共享数据模型(意图/角色/知识块)
│   ├── main.py                   # FastAPI 网关入口(挂载 Web UI)
│   ├── assistant/                # ★ Assistant 调度核心
│   │   ├── graph.py              #   LangGraph 编排:意图→分层路由→记忆
│   │   ├── intent.py             #   意图识别(qwen3.5 + 关键词兜底)
│   │   ├── memory.py             #   短期窗口 + LLM 摘要长期记忆
│   │   ├── mcp_client.py         #   MCP Client(langchain-mcp-adapters)
│   │   ├── a2a_client.py         #   A2A Client(Agent Card 发现/message.send)
│   │   └── router.py             #   /api/chat 统一入口
│   ├── rag/                      # ★ RAG 知识底座
│   │   ├── embeddings.py         #   bge-m3 (Ollama /api/embed)
│   │   ├── vectorstore.py        #   Milvus Lite collection/索引/检索
│   │   ├── bm25.py               #   BM25 稀疏检索(jieba 分词)
│   │   ├── reranker.py           #   bge-reranker-v2-m3 重排
│   │   ├── retriever.py          #   混合检索(向量+BM25→RRF→Rerank)
│   │   └── ingest.py             #   解析(txt/md/pdf/docx/视频字幕)→切分→入库
│   ├── mcp_servers/              # ★ MCP 工具层(业务系统封装)
│   │   ├── hr_server.py          #   HR 工单系统(:8001/mcp)
│   │   └── finance_server.py     #   财务报销系统(:8002/mcp)
│   ├── agents/                   # ★ A2A 专业智能体
│   │   ├── finance_agent/        #   agent_card / executor / server(:9002)
│   │   └── hr_agent/             #   agent_card / executor / server(:9001)
│   └── security/                 # ★ 安全治理
│       ├── auth.py               #   角色→工具/Agent 白名单
│       ├── audit.py              #   全链路审计 JSONL(trace_id 串联)
│       └── masking.py            #   身份证/银行卡/手机号/金额脱敏
├── scripts/
│   ├── ingest_knowledge.py       # 知识库构建脚本
│   └── demo_reimburse.py         # 端到端 demo:我要报销
├── data/knowledge/               # 样例语料(制度 md + 培训视频字幕 srt)
├── web/index.html                # Web 聊天界面
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml        # 一键启动
├── requirements.txt
└── .env.example
```

## 快速开始

前置:本地 Ollama 已启动并拉取模型(含多模态图片解析模型):

```bash
ollama pull qwen3.5
ollama pull bge-m3
ollama pull dengcao/bge-reranker-v2-m3
ollama pull qwen3-vl        # 图片/截图解析需要
```

### Docker Compose 启动(推荐)

```bash
cp .env.example docker/.env
# 编辑 docker/.env: 填入 MYSQL_PASSWORD(该文件已被 .gitignore 排除, 不会提交)
docker compose -f docker/docker-compose.yml up -d --build
# 构建知识库(可选; 现在也可通过 Web 上传)
docker compose -f docker/docker-compose.yml exec assistant python -m scripts.ingest_knowledge --dir /data/knowledge
# Web 聊天: http://localhost:8000   文档管理: http://localhost:8000/upload
```

### 本地开发

```bash
# 建议用 uv 管理依赖
pip install uv
uv sync

# 配置数据库密码: 在项目根目录 .env(或 docker/.env)添加 MYSQL_PASSWORD=...
# 也可直接设置环境变量: $env:MYSQL_PASSWORD="..." (PowerShell)

# 1. 建表自检
uv run python -m scripts.init_db

# 2. 启动业务 MCP / A2A 服务(按需)
uv run python -m app.mcp_servers.hr_server &        # :8001
uv run python -m app.mcp_servers.finance_server &   # :8002
uv run python -m app.agents.hr_agent.server &       # :9001
uv run python -m app.agents.finance_agent.server &  # :9002

# 3. 启动 Assistant 网关
uv run uvicorn app.main:app --port 8000

# Web 聊天: http://localhost:8000
# 文档上传/管理: http://localhost:8000/upload
# 命令行方式构建知识库(首次或批量):
uv run python -m scripts.ingest_knowledge --dir ./data/knowledge
```

## 端到端链路("我要报销")

1. `POST /api/chat` → Assistant 载入会话记忆(短期窗口 + 长期摘要)
2. `qwen3.5` 意图识别 → `agent_delegate / finance`
3. 权限校验(角色白名单)→ A2A Client 拉取 Finance_Agent 的 Agent Card 并 `message/send`
4. Finance_Agent(LangGraph ReAct + qwen3.5)追问/补齐要素后,经 MCP 调用
   `create_reimbursement` 创建报销单
5. 单号/审批节点沿 A2A 返回 → Assistant 回复用户;全程写 `logs/audit.jsonl`
   (同一 trace_id),敏感字段(金额/证件号/手机号)脱敏。

K8s 部署:将 `docker/docker-compose.yml` 中 5 个服务各映射为 Deployment+Service
(compose 可用 `kompose convert` 直接转换),Ollama 建议独立部署为推理服务,
`OLLAMA_BASE_URL` 指向其集群内 Service 地址即可,应用代码无需改动。
