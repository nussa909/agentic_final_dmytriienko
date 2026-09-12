import json, asyncio
from mas_langraph import init_session, process_message
 
# ── Тест-кейси ──────────────────────────────────────────────────
EVAL_SCENARIOS = [
    {
        'id': 'EVAL-01',
        'query': 'Мені 35 років, вага 82 кг, зріст 178 см. Часто болить голова. Порахуй BMI та дай рекомендації.',
        'expected_keywords': {'bmi',"надлишкова вага", "стрес, мігрень", "відпочинок"},
        'agent': {'supervisor','consultant'},
    },
    {
        'id': 'EVAL-02',
        'query': 'Скільки парацеталому дати дитині (вік - 12 років, вага - 40 кг)?',
        'expected_keywords': {'парацетамол'},
        'agent': {'supervisor','pharmacist'},
    },
    {
        'id': 'EVAL-03',
        'query': 'Який сучасний клінічний протокол лікування гострого бронхіту у дорослих?',
        'expected_keywords': {'Антибіотики', 'нестероїдні протизапальні засоби', 'moh_ukraine_protocols'},
        'agent': {'supervisor','researcher'},
    },
    {
        'id': 'EVAL-04',
        'query': 'У чоловіка діагностован бронхіт, симптоми: кашель, температура. Знайди протокол лікування бронхіту та розрахуй дозу парацетамолу (вік - 35років, вага - 75кг, зріст - 180см).',
        'expected_keywords': {'Антибіотики', 'нестероїдні протизапальні засоби', 'moh_ukraine_protocols', 'парацетамол','мг'},
        'agent': {'supervisor', 'researcher','pharmacist'},
    },
    {
        'id': 'EVAL-05',
        'query': 'Симптоми: головний біль. Жінка, вік - 35 років, вага - 65 кг, зріст - 160 см. Розрахуй мій ІМТ, подивись рекомендації щодо цього симптому та знайди в медичних довідниках "червоні прапорці" для головного болю, які вимагають МРТ.',
        'expected_keywords': {'Червоні прапорці', 'головний біль', 'medical_reference_book', 'головний біль','ІМТ'},
        'agent': {'supervisor', 'researcher','consultant'}
    }
]
 
# ── Запуск тестів ────────────────────────────────────────────────
async def run_evals(app, initial_state_fn, final_state_fn):
    results = []
    passed = 0

    for scenario in EVAL_SCENARIOS:

        try:
            config = await initial_state_fn(
                app = app,
                session_id=f"session-{scenario['id']}",
                user_prompt=scenario["query"]
            )
            result_text = await final_state_fn(app, config)

            answer = result_text.lower()
            keywords_found = [kw for kw in scenario ["expected_keywords"] if str(kw).lower() in answer]
            ok = len(keywords_found) > 0
            if ok: 
                passed += 1

            results.append({
                "id":scenario["id"],
                "query": scenario["query"],
                "expected_agent": list(scenario["agent"]),
                "keywords_found": keywords_found,
                "passed": ok,
                "answer_preview": result_text
            })

            print(f"{'OK' if ok else 'FAIL'} { scenario ['id']}: {scenario['query']}")
        except Exception as e:
            results.append({"id":scenario["id"], "passed": False, "error": str(e)})
            print(f"ERROR {scenario['id']}: {e}")

    pass_rate = passed / len(EVAL_SCENARIOS)
    summary = {
        "total": len(EVAL_SCENARIOS),
        "passed": passed,
        "failed": len(EVAL_SCENARIOS) - passed,
        "pass_rate": round(pass_rate, 2),
        "results": results
    }

    with open('eval_results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nPass-rate: {passed/len(EVAL_SCENARIOS)} = {pass_rate:.0%}")
    return summary

