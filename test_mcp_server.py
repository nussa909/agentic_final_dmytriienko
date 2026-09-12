import asyncio
import json
from mcp_server import mcp

async def call_tool(name: str, args: dict) -> str:
    """Хелпер: виклик MCP-tool у тестах."""

    result = await mcp.call_tool(name, args)

    if isinstance(result, str):
        return result
        
    if hasattr(result, 'content') and isinstance(result.content, list) and len(result.content) > 0:
        return result.content[0].text
        
    if isinstance(result, list) and len(result) > 0:
        item = result[0]
        if hasattr(item, 'text'):
            return item.text
        elif isinstance(item, dict):
            return item.get('text', str(item))
        elif isinstance(item, list) and len(item) > 0:
            sub_item = item[0]
            if hasattr(sub_item, 'text'):
                return sub_item.text
            elif isinstance(sub_item, dict):
                return sub_item.get('text', str(sub_item))
                
    return str(result)

async def main():
    print("--- Початок тестування MCP Сервера ---")

    # list_tools
    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {'symptom_lookup', 'drug_info', 'bmi_calculator', 'calorie_calculator', 'knowledge_search', 'dosage_recommendation'}.issubset(names)
    print('TEST 1 PASS — list_tools')

    # symptom_lookup — valid
    res = await call_tool('symptom_lookup', {'symptom': 'головний біль'})
    assert 'стрес, мігрень' in res
    print('TEST 2 PASS — symptom_lookup(valid)')

    # symptom_lookup — invalid format
    res = await call_tool('symptom_lookup', {'symptom': 'кашель 123'})
    assert 'Помилка формату' in res
    print('TEST 3 PASS — symptom_lookup(invalid format) -> error')

    # drug_info — valid
    res = await call_tool('drug_info', {'drug_name': 'парацетамол'})
    assert 'Анальгетики та антипіретики' in res
    print('TEST 4 PASS — drug_info(valid)')

    # drug_info  — invalid format
    res = await call_tool('drug_info', {'drug_name': 'парацетамол  987'})
    assert 'Помилка формату' in res
    print('TEST 5 PASS — drug_info(invalid format) -> error')

    # bmi_calculator — valid 
    res = await call_tool('bmi_calculator', {'weight_kg': 70.0, 'height_cm': 175.0})
    assert '22.8' in res or '22.9' in res
    print('TEST 6 PASS — bmi_calculator(valid)')

    # bmi_calculator — invalid
    res = await call_tool('bmi_calculator', {'weight_kg': 0, 'height_cm': 0})
    assert 'Помилка формату' in res or 'error' in res
    print('TEST 7 PASS — bmi_calculator(invalid format) -> error')

    # calorie_calculator — valid
    res = await call_tool('calorie_calculator', {
        'age': 30, 'weight': 70.0, 'height': 175.0, 
        'activity_level': 'moderate', 'gender': 'male'
    })
    assert '1649' in res or '1650' in res
    assert '255' in res
    print('TEST 8 PASS — calorie_calculator(valid)')

    # calorie_calculator — invalid validation (negative age)
    res_text = await call_tool('calorie_calculator', {
        'age': -5, 'weight': 70.0, 'height': 175.0, 
        'activity_level': 'moderate', 'gender': 'male'
    })
    assert 'Помилка' in res_text or 'error' in res_text
    print('TEST 9 PASS — calorie_calculator(invalid validation) → error')

    # knowledge_search — invalid
    res = await call_tool('knowledge_search', {'query': 'qq'})
    assert 'Помилка формату' in res
    print('TEST 10 PASS — knowledge_search(invalid format) -> error')

    # dosage_recommendation — invalid
    res = await call_tool('dosage_recommendation',{'drug': 'qq123', 'weight_kg': 0, 'age': 0})
    assert 'Помилка формату' in res
    print('TEST 11 PASS — knowledge_search(invalid format) -> error')

    print("--- Всі тести пройдено успішно! ---")

if __name__ == "__main__":
    asyncio.run(main())