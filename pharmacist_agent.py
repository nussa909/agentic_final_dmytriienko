from langchain_mcp_adapters.client import MultiServerMCPClient
import asyncio
from langchain_core.messages import HumanMessage, ToolMessage
import os
import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from trajectory_logger import logger
from mcp_client import MedicalMCPClient

from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv
load_dotenv()

# ── 1. Внутрішній стан Фармацевта ────────────────────────────────
# Достатньо лише масиву повідомлень, який буде синхронізуватися з глобальним MASState

AGENT_NAME = 'pharmacist'

class PharmacistState(TypedDict):
    messages: Annotated[list, operator.add]

# ── 2. Фабрика саб-графа Фармацевта ──────────────────────────────
def build_pharmacist_graph(tools: list):
    """
    Збирає саб-граф Фармацевта.
    Отримує ін'єкцію виключно ризикованих інструментів (наприклад, dosage_recommendation).
    """
    PREFIX = 'Pharmacist'
    # Ініціалізуємо LLM
    llm = ChatOpenAI(
        model='google/gemini-2.5-flash',
        temperature=0,
        api_key=os.environ.get('OPENROUTER_API_KEY', ''),
        base_url="https://openrouter.ai/api/v1"
    )

    # Біндимо ризиковані інструменти
    print(f"[{PREFIX}] tools:", [tool.name for tool in tools])
    #llm_with_tools = llm.bind_tools(tools, tool_choice="required")

    # ── Вузли ──
    def pharmacist_agent_node(state: PharmacistState) -> dict:
        """Аналізує запит Консультанта і викликає калькулятор дозування."""

        sys_msg = SystemMessage(content=(
            "Ти — професійний медичний Фармацевт. Твоя єдина мета — розрахувати дозування препарату для пацієнта.\n\n"
            "Щоб зробити розрахунок, ти маєш використати наданий тобі інструмент dosage_recommendation.\n"
            "Якщо користувач вказав назву препарату, вагу та вік (навіть якщо вони вказані у дужках) — просто використай інструмент.\n"
            "Не пиши жодного коду чи пояснень. Якщо ваги або віку не вистачає, чемно попроси користувача їх уточнити."
            "Коли інструмент повернув результат (чорновик, затверджений лікарем), ти ОБОВ'ЯЗКОВО маєш написати фінальну, ввічливу текстову відповідь для пацієнта, озвучивши цю затверджену дозу."
            "СУВОРЕ ПРАВИЛО: НЕ відмовляй у розрахунку через застереження щодо самолікування, оскільки система має Human-in-the-Loop, і ВСІ твої розрахунки обов'язково перевіряє та затверджує живий лікар-експерт."
        ))
        messages = state['messages']
        
        clean_messages = []
        for msg in messages:
            if msg.type == "human":
                clean_messages.append(msg)
            
            elif msg.type == "ai" and msg.content and not getattr(msg, "tool_calls", None):
                clean_messages.append(msg)
                
            elif msg.type == "ai" and getattr(msg, "tool_calls", None):
                # Пропускаємо, тільки якщо модель викликала 'knowledge_search'
                if any(tc.get("name") == "dosage_recommendation" for tc in msg.tool_calls):
                    clean_messages.append(msg)
                    
            elif msg.type == "tool" and msg.name == "dosage_recommendation":
                clean_messages.append(msg)

        if messages and messages[-1].type == "tool":
            llm_with_tools = llm.bind_tools(tools)
        else:
            llm_with_tools = llm.bind_tools(tools, tool_choice="dosage_recommendation")

        messages_for_llm = [sys_msg] + clean_messages
        response = llm_with_tools.invoke(messages_for_llm)

        tool_calls = []
        if response and hasattr(response, "tool_calls"):
            tool_calls = [tc["name"]
                          for tc in getattr(response, 'tool_calls', [])]

        print(
            f"\n[{PREFIX}] Аналіз запиту. Викликані інструменти: {tool_calls}")
        print(f"[{PREFIX}] response: {response.content}")

        logger.log_step(
            agent_name=AGENT_NAME,
            step_num=len(state["messages"]),
            node="pharmacist_agent_node",
            input_data=state["messages"],
            output_data=response.content,
            tool_calls=tool_calls
        )
        logger.save('trajectory.json')

        return {"messages": [response]}

    tools_node = ToolNode(tools)
    memory = MemorySaver()

    # ── Роутер ──
    def route_pharmacist(state: PharmacistState) -> Literal["tools", "__end__"]:
        """Визначає, чи LLM викликала інструмент, чи згенерувала фінальний текст."""
        last_message = state["messages"][-1]

        if getattr(last_message, "tool_calls", None):
            return "tools"
        return END

    # ── Збирання графа ──
    workflow = StateGraph(PharmacistState)

    workflow.add_node("agent", pharmacist_agent_node)
    workflow.add_node("tools", tools_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_pharmacist)

    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=memory, interrupt_after=["tools"])


# async def main():
#     client = MedicalMCPClient()
#     print("Підключено до MCP сервера...")

#     # Отримуємо тули
#     safe_tools, risky_tools, rag_tools = await client.get_categorized_tools()

#     # Будуємо ТІЛЬКИ графа Фармацевта
#     pharmacist_app = build_pharmacist_graph(risky_tools)

#     config = {'configurable': {'thread_id': 'pharmacist-direct-test-01'}}

#     # Чіткий запит з усіма даними
#     query = "Скільки парацетамолу дати дитині (12 років, 40 кг)?"

#     print("\n--- ЗАПУСК ФАРМАЦЕВТА НАПРЯМУ ---")
#     await pharmacist_app.ainvoke({"messages": [HumanMessage(content=query)]}, config)

#     # Отримуємо стан після зупинки
#     state = await pharmacist_app.aget_state(config)

#     print("\n=== РЕЗУЛЬТАТИ ===")
#     if state.next:
#         last_msg = state.values['messages'][-1]

#         if isinstance(last_msg, ToolMessage):
#             print("\n" + "="*50)
#             print("ПОТРІБНА УЧАСТЬ ЛІКАРЯ (HITL)")
#             print("="*50)

#             # Виводимо чорновик від інструмента
#             # (MCP іноді повертає контент у вигляді списку словників)
#             content = last_msg.content
#             if isinstance(content, list):
#                 content = content[0].get('text', str(content))
#             print(content)

#             print("="*50)
#             action = input(
#                 "\nВведіть 'y' для підтвердження або напишіть своє дозування: ")

#             if action.strip().lower() not in ['y', 'yes', 'т', 'так']:
#                 # Створюємо нове повідомлення від інструменту,
#                 # використовуючи ТОЙ САМИЙ ID, але з новим текстом лікаря
#                 updated_tool_message = ToolMessage(
#                     content=f"ЗАТВЕРДЖЕНО ЛІКАРЕМ (Виправлено ручками): {action}",
#                     tool_call_id=last_msg.tool_call_id
#                 )

#                 # Перезаписуємо результат інструменту в пам'яті графа!
#                 await pharmacist_app.aupdate_state(
#                     config,
#                     {"messages": [updated_tool_message]},
#                     as_node="tools"  # Вказуємо, що ми оновлюємо вузол інструментів
#                 )
#                 print("Дозування змінено і примусово збережено в систему.")
#             else:
#                 print("Дозування підтверджено без змін.")

#             # Знімаємо граф з паузи і даємо йому допрацювати
#             print("\nПродовження роботи системи...")
#             await pharmacist_app.ainvoke(None, config)

#             # Дивимося фінальну відповідь агента
#             final_state = await pharmacist_app.aget_state(config)
#             final_msg = final_state.values['messages'][-1]
#             print("\n=== ФІНАЛЬНА ВІДПОВІДЬ ПАЦІЄНТУ ===")
#             print(final_msg.content)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())
