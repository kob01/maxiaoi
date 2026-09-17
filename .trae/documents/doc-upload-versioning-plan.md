# 文档上传与知识库增强改造 · 实施计划

## 1. 概要

为「马小i」新增**文档上传入口页面 + 多模态解析入库 + LLM 自动标签 + MySQL 元数据持久化 + 章节感知父子块切片（含页码）**，聊天命中文档时给出完整章节级上下文的回答。

**重要范围约束（用户最新指示）：去掉全部版本功能，只认一个版本。**
- 无版本号、无版本历史表、无 diff 对比、无 change_summary、无版本弹窗
- 同名同类型文件再次上传 = **静默覆盖更新**（delete + 重新入库），接口返回 `overwritten: true` 供前端提示

用户已确认的决策：
- 图片解析：**Ollama 视觉模型 `qwen3-vl` 生成文字描述**（纯内网可部署）
- MySQL 密码：**仅服务启动时终端 getpass 输入**，进程内存驻留，不落盘、不写 env
- 文件身份：**文件名+扩展名一致即同一文件**（sha1(name+ext)），重传覆盖

## 2. 现状分析

| 维度 | 现状 | 差距 |
|---|---|---|
| 上传入口 | 无上传 API/页面，仅 [ingest_knowledge.py](file:///d:/ai/mxi/scripts/ingest_knowledge.py) 批量扫描目录 | 需新增上传 API + 上传页面 |
| 解析能力 | [ingest.py](file:///d:/ai/mxi/app/rag/ingest.py) 支持 txt/md/pdf/docx + srt/vtt | 缺 pptx/xlsx/图片 |
| 文件身份 | `_doc_id(path)` = sha1(文件路径) | 需改为 sha1(文件名+扩展名)，路径无关 |
| 重复上传 | `delete_by_doc` 直接覆盖 | 语义保留，补充 MySQL 元数据同步覆盖 |
| 切片质量 | 纯滑动窗口 512/64，无章节/页码/父子结构，检索命中片段上下文残缺 | 需章节感知父子块切片 + 页码记录 |
| Milvus schema | chunk_id/doc_id/title/content/source/modality | 需加 `parent_id`/`is_parent`/`page_no`/`section`（重建 collection） |
| 元数据存储 | 无任何 SQL 库 | 新增 MySQL（SQLAlchemy 2.0 async + aiomysql） |
| 聊天回答 | kb_answer 直接拼接命中 chunk | 需按父块聚合还原完整章节，标注页码/标签 |
| 标签体系 | 无 | 需 LLM 建议标签 + 用户多选确认 |

## 3. 核心设计

### 3.1 章节感知父子块切片（本次重点改造）

**解析产出 ParsedBlock（章节块）**：`{section: str, page_no: int, text: str}`，page_no 未知为 -1。

| 类型 | 章节（section） | 页码（page_no） |
|---|---|---|
| md | 按 `#`/`##` 标题切分，记标题路径 | -1 |
| txt | 单块（无章节） | -1 |
| pdf | 页内段落 | pypdf 实际页码 |
| docx | 按 Heading 样式切分 | -1（docx 无固定分页） |
| pptx | 每页 slide 为一块，记 slide 标题 | slide 序号 |
| xlsx | 每个 sheet 为一块，记 sheet 名 | sheet 序号（块内按行批切） |
| srt/vtt | 沿用现有时间段合并 | -1（时间区间已在文本内） |
| 图片 | qwen3-vl 描述文本单块 | -1 |

**父子块切分**：
- 每个 ParsedBlock 即一个**父块**；若父块文本超过 `PARENT_MAX=1200` 字，用现有滑窗逻辑切成多个子块
- 子块 `chunk_id = {doc_id}-p{seq:04d}-c{idx:02d}`，父块 `chunk_id = {doc_id}-p{seq:04d}`
- 父子均入 Milvus（父块也生成向量，便于直接命中整节），检索时 filter `is_parent == 0` 只命中子块

**检索装配（保证回答完整性）**：
```
rerank 后 top_n 子块 → 取 parent_id 去重(保最佳 rank 序)
→ store.query_parents(parent_ids) 取父块全文
→ context 按父块组织: 《title》"章节名" 第N页 + 父块完整文本
```
回答引用的不再是残缺片段，而是完整章节；BM25 语料（collect_corpus）同样只含子块。

### 3.2 上传两阶段流程

```
阶段1 POST /api/docs/upload (multipart, file+uploader)
  → 校验扩展名/大小(50MB) → doc_key = sha1(name+ext)[:16]
  → 存 data/uploads/{doc_key}/{原始文件名}
  → 查 MySQL：是否已存在（同名同类型 → overwritten 标记）
  → 解析为 ParsedBlock 列表（图片走 qwen3-vl）
  → LLM(qwen3.5) 生成 3~5 个建议标签
  → 返回 { file_token, name, ext, overwritten, suggested_tags[], preview_len }

阶段2 POST /api/docs/ingest { file_token, tags: [...] }
  → 父子块切分 → embed → Milvus delete_by_doc + upsert
  → MySQL 事务：upsert documents（含 parsed_text/chunk_count/标签）
  → BM25 增量重建 → 审计日志（trace_id）
  → 返回 { doc_key, chunk_count, tags, overwritten }
```

### 3.3 聊天感知流程

```
kb_answer 节点:
  chunks(子块) = retriever.retrieve(query)
  父块装配（3.1）+ doc_keys → docs_service.get_meta_map() 查标签
  context: [资料i] 《差旅制度》"第三章 报销流程" 第5页 (标签: 财务,报销)\n{父块全文}
  ChatResponse.metadata.docs = [{doc_key, title, tags, pages}]
```

## 4. MySQL 表设计（启动时 create_all 自动建表）

```sql
documents(
  id BIGINT PK AI, doc_key VARCHAR(32) UNIQUE,      -- sha1(name+ext) 稳定身份
  name VARCHAR(255), ext VARCHAR(16),
  modality VARCHAR(32),                              -- text/video_transcript/image
  file_path VARCHAR(512),                            -- 上传归档路径
  parsed_text MEDIUMTEXT,                            -- 解析后纯文本(重建/排查用)
  chunk_count INT, size_bytes BIGINT,
  created_by VARCHAR(64), created_at DATETIME, updated_at DATETIME)

tags(id BIGINT PK AI, name VARCHAR(64) UNIQUE,
     source VARCHAR(16) DEFAULT 'llm',               -- llm / custom
     created_at DATETIME)

document_tags(doc_key VARCHAR(32), tag_id BIGINT, PK(doc_key, tag_id))
```

## 5. 变更明细（按文件）

### 5.1 依赖 — [pyproject.toml](file:///d:/ai/mxi/pyproject.toml)
新增：`python-pptx`、`sqlalchemy[asyncio]>=2.0`、`aiomysql`、`pillow`（图片校验）。
`openpyxl`、`python-multipart` 已存在。执行 `uv lock` 更新锁文件。

### 5.2 配置 — [config.py](file:///d:/ai/mxi/app/config.py) + [.env.example](file:///d:/ai/mxi/.env.example)
新增 Settings 字段：
- `mysql_host: str = "47.116.208.170"`、`mysql_port: int = 3306`、`mysql_user: str = "sql47_116_208_1"`、`mysql_database: str = "sql47_116_208_1"`、`mysql_connect_timeout: int = 10`
- **无 `mysql_password` 字段**（仅终端输入，`app.db.session` 模块级驻留）
- `vision_model: str = "qwen3-vl"`、`upload_dir: str = "./data/uploads"`、`upload_max_mb: int = 50`、`parent_chunk_max: int = 1200`
.env.example 追加非密配置项（不含密码）。

### 5.3 数据库层（新增 `app/db/`）
- **`app/db/session.py`**：`init_engine()` 时若密码为空则 `getpass.getpass("MySQL password: ")` 输入一次驻留内存；无 tty（Docker detached）抛明确错误。SQLAlchemy 2.0 `create_async_engine("mysql+aiomysql://...")` + `async_sessionmaker`，`pool_pre_ping=True`。
- **`app/db/models.py`**：3 张表 ORM 模型。
- **`app/db/__init__.py`**。

### 5.4 文档服务层（新增 `app/docs/`）
- **`app/docs/parsers.py`**：
  - `parse_blocks(path) -> list[ParsedBlock]` 统一入口，按类型分派：
    - txt/md：`md` 按标题切 section；txt 单块
    - pdf：逐页提取，每页按段落聚合为块，记真实页码
    - docx：按 Heading 样式分节；pptx：逐 slide（标题+正文）；xlsx：逐 sheet（行拼接）
    - srt/vtt：复用现有字幕解析
    - 图片（jpg/jpeg/png/webp/bmp）：`POST {ollama}/api/generate` 调 `qwen3-vl`，base64 图像 + prompt「详细描述图片中的文字、图表与关键业务信息」，modality=`image`
- **`app/docs/service.py`**：
  - `compute_doc_key(name, ext)`、`save_upload()`、`check_existing()`
  - `suggest_tags(text)`：LLM 生成 JSON 标签列表（失败兜底空列表）
  - `ingest_confirmed()`：调 ingest 父子块入库 + MySQL 事务（覆盖时整体 upsert）
  - `get_meta_map(doc_keys)`：供聊天节点查标签
  - `list_documents()` / `list_tags()`：供前端
- **`app/docs/router.py`**（前缀 `/api/docs`）：
  - `POST /upload`（multipart）、`POST /ingest`、`GET /`（文档列表）、`GET /tags`
  - 全程审计日志（复用 [audit.py](file:///d:/ai/mxi/app/security/audit.py)）

### 5.5 RAG 改造 — [ingest.py](file:///d:/ai/mxi/app/rag/ingest.py) / [vectorstore.py](file:///d:/ai/mxi/app/rag/vectorstore.py) / [retriever.py](file:///d:/ai/mxi/app/rag/retriever.py) / [schemas.py](file:///d:/ai/mxi/app/schemas.py)
- `_doc_id` 改为按**文件名+扩展名**哈希（路径无关），`ingest_file(path)` 签名兼容 CLI 脚本
- `KnowledgeChunk` 增加 `parent_id: str = ""`、`page_no: int = -1`、`section: str = ""`、`is_parent: bool = False`
- `split_parent_child(blocks)` 新切片函数替代裸 `split_chunks` 调用（滑窗逻辑保留为块内兜底）
- Milvus schema 增加 `parent_id VARCHAR(64)`、`is_parent INT64`、`page_no INT64`、`section VARCHAR(256)`；**collection 需重建**（2.4 Lite 不支持加列，drop+create，开发期数据可重建）
- `MilvusStore.search` 加 filter `is_parent == 0`；新增 `query_parents(parent_ids)` 按父块 id 取全文；`collect_corpus` 只收子块
- `retriever.retrieve` 不变（子块粒度）；新增 `assemble_parents(chunks)` 供 graph 调用

### 5.6 聊天改造 — [graph.py](file:///d:/ai/mxi/app/assistant/graph.py) / [prompts.py](file:///d:/ai/mxi/app/assistant/prompts.py)
- `kb_answer`：子块检索 → 父块装配 → `get_meta_map` 查标签 → context 标注章节/页码/标签；`metadata["docs"]` 返回文档信息列表
- `KB_ANSWER_PROMPT` 微调：「资料标注章节与页码，回答引用时注明出处（资料序号+页码）」
- MySQL 不可用时降级：无标签标注，聊天主链路不阻断（日志告警）

### 5.7 前端
- **新增 `web/upload.html`**：
  - 拖拽/多选上传区，支持全部扩展名
  - 上传后展示：解析预览摘要、**建议标签复选框（多选）+ 自定义标签输入**
  - 同名文件重传： toast 提示「已存在同名文档，将覆盖更新」（无需选择，直接覆盖）
  - 文档列表表格：名称、类型、标签、chunk 数、上传人、更新时间
- **修改 [web/index.html](file:///d:/ai/mxi/web/index.html)**：
  - header 增加「文档管理」入口链接
  - 回答渲染时若 `metadata.docs` 非空，回答下方追加引用来源行（文档名+标签+页码）
- **修改 [main.py](file:///d:/ai/mxi/app/main.py)**：挂载 docs router、`/upload` 路由返回 upload.html、启动事件 `init_engine()`（含 getpass）+ `create_all`

### 5.8 启动脚本
- 新增 `scripts/init_db.py`：`python -m scripts.init_db` 手动建表/连通性自检（终端输密码）
- [ingest_knowledge.py](file:///d:/ai/mxi/scripts/ingest_knowledge.py) 不改接口，内部走新 doc_key + 父子块切片

## 6. 关键假设

1. 同名同类型重传=静默覆盖，不做版本保留（用户明确指示）。
2. Docker detached 模式无法 getpass——容器部署需交互终端启动或后续迭代为 secret 挂载；本期按用户决策仅终端输入。
3. 视觉模型需用户预先 `ollama pull qwen3-vl`；未拉取时图片上传报明确错误，不影响其他类型。
4. 标签多选、允许自定义新标签（source='custom'）。
5. MySQL 故障不阻断聊天主链路（标签降级），但上传/入库链路强依赖 MySQL，失败即报错回滚。
6. docx 无固定分页概念，page_no 恒为 -1，前端不展示页码；pdf/pptx/xlsx 页码有效。
7. 父块嵌入向量仅用于直命中兜底，检索结果仍按子块排序装配（is_parent 过滤在 search 层）。

## 7. 验证步骤

1. `uv lock && uv sync`；`uv run python -c "import app.main"` 编译通过。
2. `python -m scripts.init_db`：终端输密码 → 建表成功。
3. `uvicorn app.main:app --port 8000`，GET `/api/health` → ok。
4. 上传 `差旅制度.md`（含多级标题）→ 返回建议标签 → ingest 确认 → MySQL documents 1 行、Milvus 父子块齐全、chunk 带 section。
5. 上传带页码 pdf → chunk 记录 page_no；上传 pptx → section=slide 标题、page_no=slide 号。
6. 重复上传同名文件 → `overwritten: true`，Milvus chunk 全量替换、MySQL 行更新。
7. 上传 xlsx/图片各一 → 解析入库成功（图片需 qwen3-vl）。
8. 聊天提问命中制度章节 → 回答基于父块完整章节、引用标注页码，`metadata.docs` 正确。
9. `logs/audit.jsonl` 上传/ingest 事件 trace_id 完整。

## 8. 风险与注意

- **Milvus collection 重建**：schema 变更导致存量向量需重新 ingest（开发期可接受）。
- **MySQL 远程库网络依赖**：47.116.208.170 不通时上传链路不可用（按需求如此设计；聊天降级）。
- **getpass 与 uvicorn --reload**：reload 会二次提示密码，开发时建议不加 --reload 或接受重复输入。
- **图片解析耗时**：视觉模型推理慢，上传接口设 180s 超时并在前端显示解析进度态。
- **父子块存储膨胀**：父块全文冗余存储，chunk 量约 1.3~1.8 倍，SME 规模语料可接受。
