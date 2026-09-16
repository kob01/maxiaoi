"""HR_Agent A2A server entrypoint.

Run:
    python -m app.agents.hr_agent.server           # http://0.0.0.0:9001
"""

from __future__ import annotations

import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from app.agents.hr_agent.agent_card import build_agent_card
from app.agents.hr_agent.executor import HRAgentExecutor


def create_app():
    """Assemble the A2A Starlette application for HR_Agent."""
    request_handler = DefaultRequestHandler(
        agent_executor=HRAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=build_agent_card(),
        http_handler=request_handler,
    )
    return server.build()


app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9001)
