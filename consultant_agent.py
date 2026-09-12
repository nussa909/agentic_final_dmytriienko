import os
import operator
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent

from trajectory_logger import logger

from dotenv import load_dotenv
load_dotenv()

AGENT_NAME = 'consultant'

# ── 1. Схеми для структурованого виводу ─────────────────────────
class Plan(BaseModel):
    """План Консультанта для аналізу стану пацієнта."""
    steps: list[str] = Field(description='Логічні кроки для оцінки стану або підготовки відповіді')

class ReplanDecision(BaseModel):
    """Рішення після виконання кроку: продовжити, перепланувати або завершити."""
    action: Literal['continue', 'replan', 'finish'] = Field(
        description="continue=наступний крок, replan=змінити план, finish=завершити роботу Консультанта"
    )
    updated_steps: list[str] | None = Field(default=None, description="Нові кроки (якщо replan)")
    reasoning: str = Field(description="Чому прийнято таке рішення")
    final_output: str | None = Field(
        default=None, 
        description="Формується ТІЛЬКИ якщо action='finish'. Текст для пацієнта АБО запит до Супервізора на залучення Фармацевта/Дослідника."
    )

# ── 2. Внутрішній стан Консультанта ──────────────────────────────
class ConsultantState(TypedDict):
    messages: Annotated[list, operator.add] 
    plan: list[str]
    current_step: int
    results: list[str]
    completed: bool

# ── 3. LLM ───────────────────────────────────────────────────────
llm = ChatOpenAI(
    model='google/gemini-2.5-flash',
    temperature=0,
    api_key=os.environ.get('OPENROUTER_API_KEY', ''),
    base_url="https://openrouter.ai/api/v1"
)

planner_llm = llm.with_structured_output(Plan)
replanner_llm = llm.with_structured_output(ReplanDecision)

# ── 4. Вузли (Nodes) ─────────────────────────────────────────────
def consultant_planner(state: ConsultantState) -> dict:
    """Генерує план дій для обробки медичного запиту."""
    PREFIX = "Consultant Planner"
    history = "\n".join([m.content if isinstance(m.content, str) else str(m.content) for m in state['messages'][-3:]])
    
    prompt = f"""Ти — Головний Медичний Консультант. Склади план обробки запиту.
    Список доступних інструментів: 'symptom_lookup', 'drug_info', 'bmi_calculator', 'calorie_calculator'.
    Твоя задача скласти детальний і зрозумілий план обробки запиту. 
    
    ПРАВИЛА:
    - Завжди створюй максимально конкретні і зрозумілі кроки - вказуй який інструмент використовувати(якщо необхідно), вказав параметри. Все це обов'язково треба вписувать прямо в текст кроку. 
    - Якщо потрібен розрахунок дозування або глибокий пошук протоколів, медичних довідників — не намагайся зробити це сам! Останнім кроком плану вкажи: "Передати запит Супервізору для залучення Фармацевта/Дослідника".
    - Якщо інформації достатньо, останній крок: "Сформувати фінальну рекомендацію з дисплеймером".
    
    Останні повідомлення:
    {history}
    """
    
    plan = planner_llm.invoke(prompt)
    print(f"\n[{PREFIX}] План складено ({len(plan.steps)} кроків): {plan.steps}")

    logger.log_step(
        agent_name=AGENT_NAME,
        step_num=len(state["messages"]),
        node='consultant_planner',
        input_data=history[:200],
        output_data=f"Створено план: {plan.steps}"
    )
    logger.save('trajectory.json')
    
    return {"plan": plan.steps, "current_step": 0, "results": []}

async def consultant_executor(state: ConsultantState, tools: list) -> dict:
    """Виконує поточний крок плану за допомогою ReAct агента."""

    PREFIX = "Consultant Executor"
    plan = state['plan']
    step = state['current_step']
    
    if step >= len(plan):
        return {}
        
    current_task = plan[step]
    
    executor_agent = create_react_agent(llm, tools=tools)
    
    print(f"[{PREFIX}] Виконую крок {step + 1}: {current_task}")
    
    prompt = f"Виконай це завдання, використовуючи свої інструменти (якщо потрібно): {current_task}\n\nПопередні результати:\n{state.get('results', [])}"
    
    # Викликаємо ReAct агента
    result = await executor_agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
    final_answer = result["messages"][-1].content

    used_tools = []
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            used_tools.extend([tc["name"] for tc in msg.tool_calls])

    logger.log_step(
        agent_name=AGENT_NAME,
        step_num=step + 1,
        node='consultant_executor',
        input_data=current_task,
        output_data=final_answer,
        tool_calls=list(set(used_tools))
    )

    logger.save('trajectory.json')
    return {
        "results": [f"Крок {step + 1}: {final_answer}"],
        "current_step": step + 1
    }

def consultant_replanner(state: ConsultantState) -> dict:
    """Оцінює результати і вирішує, що робити далі."""

    PREFIX = "Consultant Replanner"
    prompt = f"""Ти — Репланувальник Консультанта. Оціни результати виконання кроку.
    Оригінальний план: {state['plan']}
    Поточний крок: {state['current_step']}
    Результати: {state['results']}
    
    ПРАВИЛА ОЦІНКИ:
    1. Якщо план завершено і даних достатньо, поверни action='finish' та напиши фінальну відповідь для пацієнта (з медичним дисплеймером). Фінальна відповідь має включати всі отримані результати (ІМТ, розрахунок калорій тощо) та знайдені рекомендації.
    2. Якщо ти розумієш, що запит стосується дозування (а це робота Фармацевта), поверни action='finish' і напиши у final_output: "Цей запит стосується дозування. Будь ласка, зверніться до Фармацевта." (Супервізор побачить це і перенаправить наступний запит).
    3. Якщо ти розумієш, що запит стосується глибокого пошуку в медичних протоколах, довідниках або статтях, поверни action='finish' і напиши у final_output: "Цей запит стосується дослідження. Будь ласка, зверніться Дослідника." (Супервізор побачить це і перенаправить наступний запит).
    4. Інакше — 'continue' або 'replan'.
    """
    
    decision = replanner_llm.invoke(prompt)
    print(f"[{PREFIX}] Рішення: {decision.action} ({decision.reasoning})")

    logger.log_step(
        agent_name=AGENT_NAME,
        step_num=state['current_step'],
        node='consultant_replanner',
        input_data=str(state['results'][-1:]) if state.get('results') else "Немає результатів",
        output_data=f"Action: {decision.action}. Reason: {decision.reasoning}"
    )
    logger.save('trajectory.json')
    
    if decision.action == 'finish':
        out_text = decision.final_output if decision.final_output else "Запит передано."
        return {"messages": [AIMessage(content=out_text)], "completed": True}
        
    if decision.action == 'replan' and decision.updated_steps:
        return {"plan": decision.updated_steps, "current_step": 0}
        
    return {}

def route_consultant(state: ConsultantState) -> Literal["executor", "__end__"]:
    """
    Визначає, куди має перейти виконання після вузла replanner:
    на наступний крок (executor) чи завершити роботу саб-графа (__end__).
    """

    PREFIX = "Consultant Router"
    plan = state.get("plan", [])
    current_step = state.get("current_step", 0)
    messages = state.get("messages", [])

    if current_step >= len(plan):
        print(f"[{PREFIX}] План виконано. Завершення.")
        return END

    if state.get("completed", False):
        print(f"[{PREFIX}] Отримано сигнал completed. Завершення.")
        return END

    if messages:
        last_message = messages[-1]

        if isinstance(last_message, AIMessage) and last_message.content:
            print(f"[{PREFIX}] Знайдено фінальну відповідь. Завершення.")
            return END
        
    print(f"[{PREFIX}] Перехід до кроку {current_step + 1}...")
    return "executor"

# ── 5. Фабрика для побудови саб-графа Консультанта ─────────────────
def build_consultant_graph(tools: list):
    """Збирає саб-граф Консультанта і передає йому безпечні інструменти."""
    print(f"[Consultant] tools: {[tool.name for tool in tools]}")

    async def executor_wrapper(state: ConsultantState):
        return await consultant_executor(state, tools)
        
    workflow = StateGraph(ConsultantState)
    
    workflow.add_node("planner", consultant_planner)
    workflow.add_node("executor", executor_wrapper)
    workflow.add_node("replanner", consultant_replanner)
    
    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "executor")
    workflow.add_edge("executor", "replanner")
    
    workflow.add_conditional_edges("replanner", route_consultant)

    return workflow.compile()

    