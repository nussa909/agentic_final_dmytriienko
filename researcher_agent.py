import os
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from trajectory_logger import logger

from dotenv import load_dotenv
load_dotenv()

AGENT_NAME = 'researcher'
# ── 1. Внутрішній стан Дослідника ────────────────────────────────
class ResearcherState(TypedDict):
    messages: Annotated[list, operator.add]

# ── 2. Фабрика саб-графа Дослідника ──────────────────────────────
def build_researcher_graph(tools: list):
    """Збирає RAG-агента Дослідника з підключеною векторною базою."""

    PREFIX = 'Researcher'
    print(f"[{PREFIX}] tools: {[tool.name for tool in tools]}")

    llm = ChatOpenAI(
        model='google/gemini-2.5-flash',
        temperature=0,
        api_key=os.environ.get('OPENROUTER_API_KEY', ''),
        base_url="https://openrouter.ai/api/v1"
    )

    # Біндимо інструменти до LLM
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    # ── Вузли ──
    async def researcher_agent_node(state: ResearcherState) -> dict:
        sys_msg = SystemMessage(content=(
            "Ти — Медичний Дослідник. Виконуй семантичний пошук у векторній базі "
            "для надання об'єктивних фактів та протоколів.\n\n"
            "СУВОРІ ПРАВИЛА:\n"
            "1. Завжди використовуй інструменти пошуку.\n"
            "2. Структуруй знайдену інформацію.\n"
            "3. Вказуй джерела (source), якщо вони є у JSON-відповіді.\n"
            "4. Не роби власних висновків щодо лікування."
        ))
        
        messages_for_llm = [sys_msg] + state["messages"]
        response = await llm_with_tools.ainvoke(messages_for_llm)
        
        print(f"\n[{PREFIX}] Пошук інформації. Викликані інструменти: {[tc['name'] for tc in getattr(response, 'tool_calls', [])]}")

        tool_calls = []
        if response and hasattr(response, "tool_calls"):
            tool_calls = [tc["name"]
                          for tc in getattr(response, 'tool_calls', [])]

        logger.log_step(
            agent_name=AGENT_NAME,
            step_num=len(state["messages"]),
            node="researcher_agent_node",
            input_data=state["messages"],
            output_data=response.content,
            tool_calls=tool_calls
        )
        logger.save('trajectory.json')
        
        return {"messages": [response]}

    tools_node = ToolNode(tools)

    async def debug_tools_node(state: ResearcherState):
            print("\n[DEBUG Tools] Починаю виклик інструменту до MCP сервера...")
            node = ToolNode(tools)
            try:
                res = await node.ainvoke(state)
                
                last_msg = res["messages"][-1].content
                print(f"[DEBUG Tools] Відповідь інструменту: {last_msg}")

                return res
            except Exception as e:
                print(f"[DEBUG Tools] ❌ Помилка в інструменті: {e}")
                raise e
            
    # ── Роутер ──
    def route_researcher(state: ResearcherState) -> Literal["tools", END]:
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):   
            return "tools"
        return END
        
    # ── Збирання графа ──
    workflow = StateGraph(ResearcherState)
    
    workflow.add_node("agent", researcher_agent_node)
    workflow.add_node("tools", debug_tools_node)
    
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_researcher)
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()