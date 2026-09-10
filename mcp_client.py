import os, sys
from langchain_mcp_adapters.client import MultiServerMCPClient
from guardrails import tool_guardrail
from contextlib import asynccontextmanager

class MedicalMCPClient:
    """Клієнт-обгортка для управління інструментами через MultiServerMCPClient."""

    def __init__(self, server_script: str = "mcp_server.py"):
        self.server_script = os.path.abspath(server_script)

        self.config = {
            'medical_support': { 
                'command': 'python',
                'args': [self.server_script],
                'transport': 'stdio',
                'env': os.environ.copy()
            }
        }

        # self.config = {
        #     'medical_support': {
        #         'command': sys.executable,
        #         'args': [self.server_script],
        #         'transport': 'stdio',
        #         'env': os.environ.copy()   # ВИПРАВЛЕНО: Передаємо API ключі серверу
        #     }
        # }
        self.mcp_client = MultiServerMCPClient(self.config)

    # async def __aenter__(self):
    #     await self.mcp_client.__aenter__()
    #     return self

    # async def __aexit__(self, exc_type, exc_val, exc_tb):
    #     await self.mcp_client.__aexit__(exc_type, exc_val, exc_tb)

    @asynccontextmanager
    async def active_session(self):
        """Створює і тримає з'єднання з сервером відкритим."""
        async with self.mcp_client.session('medical_support') as session:
            yield session

    # async def get_categorized_tools(self) -> tuple[list, list, list]:
    #         """Повертає три масиви інструментів"""
    #         all_tools = await self.mcp_client.get_tools()

    #         # GUARDRAIL: Tool Permissions
    #         pharmacist_tools = [t for t in all_tools if tool_guardrail('pharmacist', t.name)]
    #         consultant_tools = [t for t in all_tools if tool_guardrail('consultant', t.name)]
    #         researcher_tools = [t for t in all_tools if tool_guardrail('researcher', t.name)]
                    
    #         return pharmacist_tools, consultant_tools, researcher_tools