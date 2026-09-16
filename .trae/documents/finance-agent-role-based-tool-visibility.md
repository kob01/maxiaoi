# 财务Agent权限分级：角色×工具白名单矩阵（结构化身份 + 权限Mask + 动态System Prompt）

## Summary

为 Finance_Agent 实现按角色分级的工具可见性控制。**角色×工具采用显式白名单矩阵（默认拒绝）**，身份通过 **A2A 协议原生的 `Message.metadata` 结构化字段**传递（不从对话文本解析角色），可见性由**权限Mask（工具列表过滤，硬控制）+ 动态System Prompt（权限边界引导，软控制）**双重保证。

### 角色×工具矩阵（唯一事实来源，放在 auth.py）

| MCP 工具（finance server） | 普通员工<br>`employee` | 部门经理<br>`manager`(新增) | HR/财务专员<br>`hr`/`finance` | 管理员<br>`admin` |
|---|:---:|:---:|:---:|:---:|
| create_reimbursement（提交报销） | ✓ | ✓ | ✓ | ✓ |
| query_reimbursement（查报销单） | ✓ | ✓ | ✓ | ✓ |
| list_reimbursements（列报销单） | ✓ | ✓ | ✓ | ✓ |
| get_reimbursement_policy（政策咨询） | ✓ | ✓ | ✓ | ✓ |
| **finance_budget_query（预算查询，新增）** | **✗ 隐藏** | ✓ | ✓ | ✓ |

- **白名单语义**：工具未列入某角色的白名单 → 对该角色完全隐藏（LLM 看不见、调不到），而非"调用时报错"。
- hr server 工具本期不分级（需求仅限财务域），矩阵中对 hr 域返回全量。

## Current State Analysis

- **入口**：`app/main.py` → [router.py](file:///d:/ai/mxi/app/assistant/router.py) POST `/api/chat`，[ChatRequest](file:///d:/ai/mxi/app/schemas.py#L39-L45) 携带可信 `role`（现有 employee/hr/finance/admin，**缺 manager**），Web 下拉框（[web/index.html](file:///d:/ai/mxi/web/index.html#L34-L39)）无"经理"选项。
- **编排层**：[graph.py](file:///d:/ai/mxi/app/assistant/graph.py) `tool_execute`（L91-110）把目标 server **全部**工具直接喂给 `create_react_agent`，无工具级过滤；`agent_delegate`（L112-128）仅以文本 `[employee_id=XXX]` 传身份。
- **财务Agent**：[executor.py](file:///d:/ai/mxi/app/agents/finance_agent/executor.py) 静态 SYSTEM_PROMPT + 全量工具，agent 单例缓存，无角色感知。
- **财务MCP**：[finance_server.py](file:///d:/ai/mxi/app/mcp_servers/finance_server.py) 现有 4 工具，**无预算查询工具**。
- **权限模块**：[auth.py](file:///d:/ai/mxi/app/security/auth.py) 仅有 server 级 `MCP_WHITELIST`；`TOOL_ROLE_RESTRICTIONS` 空置；且 `tool_execute` 里 `check_mcp_permission(role, target, "*")` 实际只做 server 级校验。
- **A2A 协议能力（已核实）**：a2a-sdk 0.3.x 的 `Message` 含 `metadata: dict[str, Any] | None` 结构化字段（协议规范原生支持），服务端 executor 可经 `RequestContext.message` 读取；客户端 `MessageSendParams(message=Message(..., metadata=...))` 即可下发。**这是比文本标签更严谨的身份通道**。

## Proposed Changes

### 1. `app/mcp_servers/finance_server.py` — 新增预算查询工具

- 新增内存数据 `_BUDGETS`（部门 → 年度预算/已用：研发部、市场部、人事部、财务部）。
- 新增工具 `finance_budget_query(department: str)`：返回预算总额/已用/剩余；未知部门返回 error payload（与现有工具风格一致）。FastMCP 函数名即工具名。

### 2. `app/security/auth.py` — 角色×工具白名单矩阵（核心）

- `Role` 语义不变（manager 在 schemas.py 中新增，见第 6 条）。
- 新增**显式白名单矩阵**（即上文 Summary 中的表格，代码为唯一事实来源）：

  ```python
  # 角色 -> finance 域可见工具白名单；None 表示该域全量可见（默认拒绝,未列出的工具一律隐藏）
  FINANCE_TOOL_WHITELIST: dict[Role, set[str] | None] = {
      Role.EMPLOYEE: {"create_reimbursement", "query_reimbursement",
                       "list_reimbursements", "get_reimbursement_policy"},
      Role.MANAGER:  {"create_reimbursement", "query_reimbursement",
                       "list_reimbursements", "get_reimbursement_policy",
                       "finance_budget_query"},
      Role.HR: None, Role.FINANCE: None, Role.ADMIN: None,
  }
  ```

- 新增 `filter_tools_for_role(role, server_name, tools)`：
  - `server_name == "finance"` → 按 `FINANCE_TOOL_WHITELIST` 过滤（角色无条目/空集 → 全部隐藏）；
  - 其他 server（hr）→ 原样返回（本期不分级）。
  - 供 orchestrator 与 Finance_Agent 复用，**两条调用路径同源同矩阵**。
- `MCP_WHITELIST` / `AGENT_WHITELIST` 补充 `Role.MANAGER` 条目（沿用 employee 级别），保持 server/agent 级门禁完整。
- 约束写入代码注释：**MCP server 新增工具时必须同步维护矩阵**（默认拒绝是有意为之）。

### 3. `app/assistant/graph.py` — 编排层：结构化身份下发 + 同源过滤

- `agent_delegate`（L112-128）：
  - 保留现有文本 `[employee_id=...]`（HR_Agent 兼容依赖，本期不动 HR）；
  - **新增**：`get_a2a_pool().send(target, task, metadata={"user_id": state["user_id"], "role": state["role"].value})`，身份以协议级 metadata 结构化下发。
- `tool_execute`（L91-110）：
  - `get_tools(target)` 后经 `filter_tools_for_role(state["role"], target, tools)` 过滤再建 ReAct Agent（**硬控制**：被隐藏的工具不在 LLM 工具表中，物理不可调）；
  - 过滤后无可用工具 → 直接返回"权限不足"，不进入 Agent 循环；
  - 审计日志补充 `role`、过滤后工具名列表。

### 4. `app/assistant/a2a_client.py` — metadata 透传

- `A2AClientPool.send()` 增加可选参数 `metadata: dict[str, Any] | None = None`，构造 `Message(..., metadata=metadata)`。不传时行为不变（HR 路径零影响）。

### 5. `app/agents/finance_agent/executor.py` — 结构化身份读取 + 按角色构建（核心）

- **身份读取（严谨性关键）**：
  - `execute()` 从 `context.message.metadata` 读取 `user_id` / `role`（**唯一的角色来源**，完全忽略用户文本中可能伪造的任何 `[role=...]` 字样）；
  - metadata 缺失或非法（直连 A2A 调用者）→ `role` 回退 `employee`（最小权限）；`user_id` 回退解析文本 `[employee_id=...]`（保留现有直连兼容），再缺失则由 Agent 追问。
- `FinanceAgent.invoke(user_text, user_id, role)`；agent 缓存改为 `dict[Role, graph]` 按角色缓存。
- `_ensure_agent(role)`：MCP 工具经 `filter_tools_for_role(role, "finance", tools)` 过滤后传入 `create_react_agent`（与编排层同矩阵）。
- `_build_role_prompt(role, user_id)` — **动态 System Prompt**（替代静态常量），按角色注入：
  - 当前用户工号与权限层级（普通员工/部门经理/HR·财务专员）；
  - 本角色**实际可用**的工具清单与职责边界；
  - 越权引导：若用户请求超出权限的能力（如员工查预算），礼貌说明无权限并建议联系部门经理/财务专员，不编造、不尝试调用不存在的工具。
- 审计日志记录 `role` 与可见工具列表。

### 6. `app/schemas.py` / `web/index.html` — 角色扩充

- `Role` 枚举新增 `MANAGER = "manager"`。
- Web 角色下拉框增加 `<option value="manager">经理</option>`。

### 7. `app/agents/finance_agent/agent_card.py` — 能力补录

- skills 增加 `query_budget`（预算查询）。卡片描述全量能力供发现，角色可见性由运行时 Mask 决定。

## Assumptions & Decisions

1. **角色来源严谨性**（回应"只从上下文取 role 不严谨"）：
   - 角色只信 `Message.metadata`（编排层从可信 `ChatRequest.role` 写入，协议结构化字段，非自由文本）；
   - Agent 端**不解析**用户文本中的角色字样，伪造 `[role=admin]` 无效；
   - metadata 缺失按 `employee` 最小权限处理；
   - 本 demo 的信任锚仍是 `ChatRequest.role`（Web 下拉声明）；生产环境应替换为 SSO/JWT 解析角色，架构不变，仅替换 orchestrator 处的角色来源。
2. **默认拒绝**：白名单矩阵未覆盖的工具一律隐藏；矩阵是唯一事实来源，两处消费点（orchestrator 直连 MCP 路径、Finance_Agent A2A 路径）同源过滤，堵住旁路。
3. **范围**：本期仅财务域分级；HR 域矩阵留扩展点（新增 `HR_TOOL_WHITELIST` + `filter_tools_for_role` 分支即可）。HR_Agent 收到 metadata 不受影响（不读取）。
4. **兼容性**：`agent_delegate` 文本中的 `[employee_id=]` 保留（HR 路径依赖）；Finance_Agent 改为优先 metadata、文本仅作 user_id 回退。

## Verification

0. 前置检查：`python -c "from a2a.types import Message; print(Message.model_fields.get('metadata'))"` 确认安装的 a2a-sdk 0.3.x 支持 metadata 字段。
1. 启动服务：finance MCP(8002) → finance_agent(9002) → assistant(8000)。
2. **employee 隐藏验证**：`POST /api/chat` role=employee，"查一下研发部的预算" → 答复说明无权限/建议联系经理或财务专员；审计日志显示过滤后工具列表不含 finance_budget_query。
3. **manager 放行验证**：role=manager 同消息 → 返回研发部预算（总额/已用/剩余）。
4. **hr/finance/admin 全量验证**：role=finance 同消息 → 同 manager 结果。
5. **员工原有能力不受影响**：role=employee "我要报销500元差旅费" → 正常创建报销单。
6. **A2A 路径验证**：role=manager "帮我报销600元培训费并看看人事部预算" → intent 走 agent_delegate，Finance_Agent 按 metadata 角色调用工具成功。
7. **伪造抵抗验证**：直接 curl Finance_Agent 的 A2A 端点，正文携带 `[role=admin]` 但 metadata 无 role → 必须按 employee 处理（预算工具不可见）。
8. **旁路验证**：role=employee 走 tool_call 直连 MCP 路径问预算 → tool_execute 过滤后无可用预算工具，返回权限不足。
9. 检查 `logs/audit.jsonl`：每轮含 role、过滤后工具列表。
