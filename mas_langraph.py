import operator, os, time, asyncio

from typing import Annotated, Literal, Sequence, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage,AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# local imports
from mcp_client import MedicalMCPClient

from consultant_agent import build_consultant_graph
from pharmacist_agent import build_pharmacist_graph
from researcher_agent import build_researcher_graph

from guardrails import input_guardrail, output_guardrail, tool_guardrail, RateLimiter

from trajectory_logger import logger

from dotenv import load_dotenv
load_dotenv()

MAX_STEPS = 5
TIMEOUT = 120.0

pharmacist_tools = []
consultant_tools = []
researcher_tools = []


# ── 1. Стан Графа (State) ────────────────────────────────────────
class MASState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next_node: str
    step_count: int
    start_time: float                               

# ── 2. Структурований вивід для Супервізора ──────────────────────
class RouterDecision(BaseModel):
    """Рішення Супервізора щодо наступного кроку в системі."""
    action: Literal["consultant", "pharmacist", "researcher", "finish"] = Field(
        description="Вузол, якому передається керування, або finish для завершення роботи."
    )
    reasoning: str = Field(
        description="Коротке логічне пояснення, чому обрано саме цього агента."
    )

# ── 3. LLM та Системний Промпт Супервізора ───────────────────────
async def build_supervisor_graph(pharmacist_tools: list, consultant_tools: list, researcher_tools: list, checkpointer):
    """
    Збирає граф Cупервізора.
    """
    llm = ChatOpenAI(
        model='google/gemini-2.5-flash',
        temperature=0,
        api_key=os.environ.get('OPENROUTER_API_KEY', ''),
        base_url="https://openrouter.ai/api/v1"
    )

    supervisor_llm = llm.with_structured_output(RouterDecision)

    SUPERVISOR_PROMPT = """Ти — Головний Супервізор медичної мультиагентної системи. 
    Твоя задача — маршрутизувати запит користувача до правильного експерта.

    ЕКСПЕРТИ:
    - 'pharmacist': Вибирай його, якщо користувач просить розрахувати дозу, питає про дозування, або вказує препарат, вагу та вік для розрахунку.
    - 'researcher': Вибирай його, якщо запит вимагає глибокого пошуку по медичних протоколах, симптомах або складних хворобах.
    - 'consultant': Вибирай його для загальних медичних питань, розшифровки базових показників (наприклад, ІМТ), або якщо запит нечіткий і потребує загального аналізу.

    ПРАВИЛА: 
    Маршрутизуй напряму до потрібного експерта. Якщо експерт успішно виконав свою роботу і надав відповідь користувачу, поверни 'finish'.
    Якщо останнє повідомлення в історії написане Агентом (штучним інтелектом) і це уточнююче запитання до користувача АБО готова відповідь — ти ОБОВ'ЯЗКОВО повинен вибрати 'finish' (або відповідний статус завершення), щоб передати слово людині! Не направляй запит назад до експертів.
    """

    consultant_subgraph = build_consultant_graph(consultant_tools)
    pharmacist_subgraph = build_pharmacist_graph(pharmacist_tools)
    researcher_subgraph = build_researcher_graph(researcher_tools)

    # ── 4. Вузол Супервізора ─────────────────────────────────────────
    def supervisor_node(state: MASState) -> dict:
        messages = state['messages'][-1].content if state['messages'] else ''
        step = state.get('step_count', 0) + 1
        start_time = state.get("start_time", time.time())

        # ПЕРЕВІРКА ТАЙМАУТУ (TIMEOUT)
        elapsed_time = time.time() - start_time
        if elapsed_time > TIMEOUT:
            print(f"[SUPERVISOR] Перевищено ліміт часу ({elapsed_time:.1f}s). Примусове завершення.")
            timeout_msg = AIMessage(content="Вибачте, але розрахунок займає надто багато часу. Спробуйте, будь ласка, сформулювати запит інакше.")
            return {
                "messages": [timeout_msg], 
                "next": END, 
                "step_count": step, 
                "start_time": start_time
            }

        # ПЕРЕВІРКА МАКСИМАЛЬНОЇ КІЛЬКОСТІ КРОКІВ (MAX_STEP)
        if step > MAX_STEPS:
            print(f"🔄 [SUPERVISOR] Перевищено ліміт кроків ({step}/{MAX_STEPS}). Примусове завершення.")
            limit_msg = AIMessage(content="Система не змогла знайти однозначної відповіді та зациклилась. Будь ласка, зверніться до лікаря.")
            return {
                "messages": [limit_msg], 
                "next": END, 
                "step_count": step, 
                "start_time": start_time
            }

        prompt = [SystemMessage(content=SUPERVISOR_PROMPT)] + list(messages)
        
        decision = supervisor_llm.invoke(prompt)

        print(f"\n[SUPERVISOR] Рішення: {decision.action}")
        print(f"[SUPERVISOR] Логіка: {decision.reasoning}")

        logger.log_step('supervisor', step, 'supervisor_node', messages, decision, [])
        logger.save('trajectory.json')
        
        return {"next_node": decision.action,'step_count': step}

    # Функції-заглушки для ваших агентів
    async def consultant_node(state: MASState):
        input_msg = state["messages"][-1].content if state["messages"] else ""
        
        # Викликаємо саб-граф Консультанта, передаючи йому всю історію
        result = await consultant_subgraph.ainvoke({"messages": state["messages"]})
        
        # Витягуємо лише нові повідомлення, які згенерував саб-граф
        new_messages = result["messages"][len(state["messages"]):]
        output_msg = new_messages[-1].content if new_messages else "Немає відповіді."
        
        # Логуємо крок
        logger.log_step(
            agent_name="consultant",
            step_num=len(state["messages"]), 
            node="consultant_node",
            input_data=input_msg,
            output_data=output_msg,
            tool_calls=[]
        )
        logger.save('trajectory.json')
        
        return {"messages": new_messages}

    async def pharmacist_node(state: MASState):
        input_msg = state["messages"][-1].content if state["messages"] else ""
        
        # Викликаємо саб-граф Фармацевта. 
        # Якщо тут налаштовано interrupt_before, граф призупиниться.
        result = await pharmacist_subgraph.ainvoke({"messages": state["messages"]})
        
        new_messages = result["messages"][len(state["messages"]):]
        output_msg = new_messages[-1].content if new_messages else "Немає розрахунку."
        
        # Витягуємо виклики інструментів для логера (якщо є)
        tool_calls = []
        if new_messages and hasattr(new_messages[-1], "tool_calls"):
            tool_calls = [tc["name"] for tc in new_messages[-1].tool_calls]
        
        logger.log_step(
            agent_name="pharmacist",
            step_num=len(state["messages"]),
            node="pharmacist_node",
            input_data=input_msg,
            output_data=output_msg,
            tool_calls=tool_calls
        )
        logger.save('trajectory.json')
        
        return {"messages": new_messages}

    async def researcher_node(state: MASState):
        input_msg = state["messages"][-1].content if state["messages"] else ""
        
        # Викликаємо саб-граф Дослідника
        result = await researcher_subgraph.ainvoke({"messages": state["messages"]})
        
        new_messages = result["messages"][len(state["messages"]):]
        output_msg = new_messages[-1].content if new_messages else "Немає даних."
        
        tool_calls = []
        if new_messages and hasattr(new_messages[-1], "tool_calls"):
            tool_calls = [tc["name"] for tc in new_messages[-1].tool_calls]
        
        logger.log_step(
            agent_name="researcher",
            step_num=len(state["messages"]),
            node="researcher_node",
            input_data=input_msg,
            output_data=output_msg,
            tool_calls=tool_calls
        )
        logger.save('trajectory.json')
        
        return {"messages": new_messages}

    # ── 5. Роутер для графа ──────────────────────────────────────────
    def supervisor_router(state: MASState) -> str:
        if state["next_node"].lower() == "finish":
            return END
        return state["next_node"]

    # ── ГРАФ ──────────────────────────────────────────
    graph = StateGraph(MASState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("consultant", consultant_node)
    graph.add_node("pharmacist", pharmacist_node)
    graph.add_node("researcher", researcher_node)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges("supervisor", supervisor_router)

    graph.add_edge("consultant", "supervisor")
    graph.add_edge("pharmacist", "supervisor")
    graph.add_edge("researcher", "supervisor")

    # ── Checkpointer ──
    return graph.compile(checkpointer=checkpointer)


# ── Запуск ─────────────────────────────────────────────────────────
rate_limiter = RateLimiter(max_calls=20, window_sec=60)

async def init_session(app, session_id: str, user_prompt: str) -> dict:
    """Ініціалізує нову сесію або підключається до існуючої."""
    config = {'configurable': {'thread_id': session_id}}
    state = await app.aget_state(config)

    # GUARDRAIL: Rate Limiter
    is_allowed, rl_msg = rate_limiter.check(session_id)
    if not is_allowed:
        print(f"{rl_msg}. Зачекайте хвилину.")

    # GUARDRAIL: Input Guardrail (Ін'єкції та довжина)
    is_safe, clean_query = input_guardrail(user_prompt)
    if not is_safe:
        print(f"Блокування системи: {clean_query}")  

    if not state.values:
        print(f"Створення нової сесії: {session_id}")
        await app.ainvoke({
            'messages': [HumanMessage(content=user_prompt)],
            'next_node': [],
            'step_count': 0
        }, config=config)
    else:
        print(f"Знайдено існуючу сесію {session_id}. Підключення до чекпоінта...")

    return config


from langchain_core.messages import ToolMessage
async def process_message(app, config):
    state = await app.aget_state(config, subgraphs=True)
    
    if state.next:
        print("\n" + "="*50)
        print("👨‍⚕️ ПОТРІБНА УЧАСТЬ ЛІКАРЯ (СИСТЕМА НА ПАУЗІ)")
        print("="*50)
        
        if hasattr(state, 'tasks') and len(state.tasks) > 0:
            subgraph_task = state.tasks[0]
            subgraph_state = subgraph_task.state
            subgraph_config = subgraph_task.state.config
            
            last_msg = subgraph_state.values['messages'][-1]
            
            if getattr(last_msg, 'type', '') == 'tool':
                content = last_msg.content
                if isinstance(content, list):
                    content = content[0].get('text', str(content))
                print(content)
        
        print("="*50)
        action = input("\nВведіть 'y' для підтвердження або ваші правки: ")
        
        if action.strip().lower() not in ['y', 'yes', 'т', 'так']:
            from langchain_core.messages import ToolMessage
            updated_tool = ToolMessage(
                content=f"ЗАТВЕРДЖЕНО ЛІКАРЕМ (Виправлено: {action})",
                tool_call_id=last_msg.tool_call_id
            )
            await app.aupdate_state(subgraph_config, {"messages": [updated_tool]}, as_node="tools")
            print("✅ Правки збережено у внутрішній стан.")
        else:
            print("✅ Підтверджено без змін.")
            
        print("\nПродовжуємо роботу графа...")
        await app.ainvoke(None, config)
        
    else:
        print("\nГраф завершив роботу без пауз (HITL не знадобився).")


    final_state = await app.aget_state(config)
    final_text = final_state.values['messages'][-1].content

    # GUARDRAIL: Output Guardrail (Видалення PII)
    safe_output, pii_found = output_guardrail(final_text)
    if pii_found:
        print(f"[SYSTEM LOG] Заблоковано витік даних: {pii_found}")

    result = f"\n=== ФІНАЛЬНА ВІДПОВІДЬ ПАЦІЄНТУ ===\n {safe_output}"
    print(result)
    return result


async def main():
    from langchain_mcp_adapters.tools import load_mcp_tools

    global pharmacist_tools, consultant_tools, researcher_tools

    client = MedicalMCPClient()
    print("Підключено до MCP сервера...")
    
    # pharmacist_tools, consultant_tools, researcher_tools = await client.get_categorized_tools()
    
    async with client.active_session() as session:
        # Завантажуємо інструменти з активної сесії
        all_tools = await load_mcp_tools(session)
        
        # GUARDRAIL: Розподіляємо інструменти
        pharmacist_tools = [t for t in all_tools if tool_guardrail('pharmacist', t.name)]
        consultant_tools = [t for t in all_tools if tool_guardrail('consultant', t.name)]
        researcher_tools = [t for t in all_tools if tool_guardrail('researcher', t.name)]
        
        print("Інструменти завантажено! Запускаємо граф...")


        async with AsyncSqliteSaver.from_conn_string("checkpoints.sqlite") as saver:
            supervisor_app = await build_supervisor_graph(pharmacist_tools, consultant_tools, researcher_tools, saver)

            # print("\n--- ТЕСТ 1 ---")
            # config_1 = await init_session(
            #     app = supervisor_app,
            #     session_id='session-023',
            #     user_prompt="Мені 35 років, вага 82 кг, зріст 178 см. Часто болить голова. Порахуй BMI та дай рекомендації."
            # )
            # await process_message(supervisor_app, config_1)

            # print("\n--- ТЕСТ 2 ---")
            # config_2 = await init_session(
            #     app = supervisor_app,
            #     session_id='session-020',
            #     user_prompt="Скільки парацеталому дати дитині (вік - 12 років, вага - 40 кг)?"
            # )
            # await process_message(supervisor_app, config_2)

            # print("\n--- ТЕСТ 3 ---")
            # config_3 = await init_session(
            #     app = supervisor_app,
            #     session_id='session-010',
            #     user_prompt="Який сучасний клінічний протокол лікування гострого бронхіту у дорослих?"
            # )
            # await process_message(supervisor_app, config_3)

            print("\n--- ТЕСТ 4 ---")
            config_4 = await init_session(
                app = supervisor_app,
                session_id='session-027',
                user_prompt="У чоловіка бронхіальна астма, часто буває задишка. Знайди протокол лікування бронхіальної астми та розрахуй дозу сальбутамолу (вага - 75кг, зріст - 180см)."
            )
            await process_message(supervisor_app, config_4)

if __name__ == "__main__":
    asyncio.run(main())
