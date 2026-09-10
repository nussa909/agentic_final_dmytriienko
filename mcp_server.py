"""MCP Server для вашого домену. Запуск: python mcp_server.py (stdio)"""
import json, os, sys, logging
from datetime import datetime, timezone
from langchain_core.documents import Document
from typing import Dict, Union, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from mcp.server.fastmcp import FastMCP

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(filename='mcp_debug.log', level=logging.INFO, 
                    format='%(asctime)s - %(message)s')

mcp = FastMCP(
    name='medical_tools_server',
    instructions='Сервер з ресурсами medical support',
)

########################################### Mock-бази ####################################################
# 1. Mock-база симптомів
MOCK_SYMPTOMS = {
    "головний біль": "Можливі причини: стрес, мігрень, зневоднення, недосипання. Рекомендації: відпочинок, питний режим, за потреби - знеболювальне.",
    "кашель": "Можливі причини: ГРВІ, алергія, бронхіт. Рекомендації: тепле пиття, зволоження повітря. Якщо триває понад 2 тижні — зверніться до лікаря.",
    "лихоманка": "Підвищення температури тіла. Може свідчити про інфекцію. Рекомендації: пити багато рідини. Збивати температуру вище 38.5°C.",
    "біль у горлі": "Можливі причини: фарингіт, ангіна, ГРВІ. Рекомендації: полоскання, теплі напої, льодяники від болю в горлі.",
    "нудота": "Можливі причини: харчове отруєння, вірусна інфекція, стрес або заколисування. Рекомендації: пити воду невеликими порціями, дієта BRAT.",
    "запаморочення": "Можливі причини: зниження тиску, зневоднення, проблеми з внутрішнім вухом. Рекомендації: сісти або лягти, випити води.",
    "біль у животі": "Може бути ознакою спазму, гастриту або отруєння. Увага: гострий біль у правому нижньому куті може свідчити про апендицит (негайний виклик швидкої).",
    "задишка": "Можливі причини: астма, тривожність, серцеві захворювання, COVID-19. Якщо виникає раптово у стані спокою — потрібна невідкладна допомога.",
    "біль у спині": "Можливі причини: перенапруження м'язів, остеохондроз, грижа. Рекомендації: уникати підняття важких предметів, легка розтяжка.",
    "безсоння": "Можливі причини: стрес, надмірне споживання кофеїну, порушення режиму. Рекомендації: гігієна сну, відмова від екранів за годину до сну.",
    "втома": "Можливі причини: перевтома, стрес, дефіцит вітамінів, анемія, порушення сну. Рекомендації: повноцінний сон, збалансоване харчування, зниження рівня стресу, консультація лікаря для здачі аналізів."
}

# 2. Mock-база препаратів
MOCK_DRUGS = {
    "парацетамол": "Група: Анальгетики та антипіретики. Покази: біль слабкої та помірної інтенсивності, лихоманка. Побічні дії: рідко алергічні реакції.",
    "ібупрофен": "Група: НПЗЗ (Нестероїдні протизапальні засоби). Покази: запалення, біль, жар. Протипоказання: виразкова хвороба шлунка.",
    "аспірин": "Група: НПЗЗ, антиагреганти. Покази: головний біль, профілактика тромбозів. Не рекомендується дітям до 16 років.",
    "лоратадин": "Група: Антигістамінні (протиалергічні). Покази: алергічний риніт, кропив'янка. Перевага: не викликає сильної сонливості.",
    "омепразол": "Група: Інгібітори протонної помпи. Покази: печія, гастрит, виразка шлунка. Приймати за 30 хв до їжі.",
    "азитроміцин": "Група: Антибіотики (макроліди). Покази: бактеріальні інфекції дихальних шляхів. Відпускається лише за рецептом.",
    "метформін": "Група: Пероральні гіпоглікемічні засоби. Покази: цукровий діабет 2 типу. Побічні дії: можливі розлади шлунково-кишкового тракту.",
    "амоксицилін": "Група: Антибіотики (пеніциліни). Покази: широкий спектр бактеріальних інфекцій. Відпускається за рецептом лікаря.",
    "диклофенак": "Група: НПЗЗ. Покази: біль у суглобах, спині, запальні процеси. Часто використовується у формі гелів або мазей.",
    "цетиризин": "Група: Антигістамінні. Покази: алергічні реакції. Може викликати легку сонливість у деяких пацієнтів.",
    "амброксол": "Група: Муколітики. Покази: вологий кашель, бронхіт. Сприяє розрідженню та виведенню мокротиння.",
    "бензидамін": "Група: Місцеві нестероїдні протизапальні засоби. Покази: біль у горлі, фарингіт, ангіна. Використовується у вигляді спрею або льодяників.",
    "дротаверин": "Група: Спазмолітики (більш відомий як Но-шпа). Покази: спазми гладкої мускулатури, біль у животі, кишкові коліки.",
    "діосмектит": "Група: Ентеросорбенти (Смекта). Покази: діарея, нудота, харчові отруєння. Зв'язує та виводить токсини з організму.",
    "мелатонін": "Група: Адаптогени, аналоги гормону сну. Покази: безсоння, порушення циркадних ритмів (наприклад, при зміні часових поясів).",
    "сальбутамол": "Група: Бронходилататори. Покази: задишка, напади бронхіальної астми. Застосовується переважно у формі інгалятора.",
    "бетагістин": "Група: Засоби, що застосовуються при вестибулярних порушеннях. Покази: запаморочення, шум у вухах, хвороба Меньєра.",
    "магній": "Група: Мінеральні добавки. Покази: втома, стрес, м'язові судоми. Часто комбінується з вітаміном В6 для кращого засвоєння."
}

MOCK_DOSAGES = {
    "парацетамол": 15.0,    # 15 мг/кг
    "ібупрофен": 10.0,      # 10 мг/кг
    "аспірин": 8.0,         # 8 мг/кг
    "лоратадин": 0.2,       # 0.2 мг/кг
    "омепразол": 0.5,       # 0.5 мг/кг
    "азитроміцин": 10.0,    # 10 мг/кг
    "метформін": 15.0,      # 15 мг/кг
    "амоксицилін": 25.0,    # 25 мг/кг
    "диклофенак": 2.0,      # 2 мг/кг
    "цетиризин": 0.2,       # 0.2 мг/кг
    "амброксол": 1.2,       # 1.2 мг/кг
    "бензидамін": 0.5,      # 0.5 мг/кг
    "дротаверин": 1.5,      # 1.5 мг/кг
    "діосмектит": 50.0,     # 50 мг/кг
    "мелатонін": 0.05,      # 0.05 мг/кг
    "сальбутамол": 0.1,     # 0.1 мг/кг
    "бетагістин": 0.5,      # 0.5 мг/кг
    "магній": 5.0           # 5 мг/кг
}

MOCK_DOCUMENTS = [
    # --- Клінічні протоколи (Protocols) ---
    Document(
        page_content="Клінічний протокол МОЗ: Лікування гострого бронхіту у дорослих. Антибіотики не рекомендовані для рутинного лікування гострого неускладненого бронхіту, оскільки в 90% випадків захворювання має вірусну етіологію. Препаратами вибору для полегшення симптомів є нестероїдні протизапальні засоби (НПЗЗ) та парацетамол.",
        metadata={"source": "moh_ukraine_protocols", "topic": "pulmonology", "type": "protocol"}
    ),
    Document(
        page_content="Міжнародні стандарти лікування артеріальної гіпертензії: Цільовий рівень артеріального тиску для пацієнтів віком до 65 років становить 120-129/70-79 мм рт. ст. Терапію першої лінії слід починати з комбінації двох препаратів: інгібітора АПФ або БРА у поєднанні з блокатором кальцієвих каналів або діуретиком.",
        metadata={"source": "esc_guidelines", "topic": "cardiology", "type": "protocol"}
    ),
    Document(
        page_content="Настанови з управління бронхіальною астмою (GINA 2023): Для пацієнтів старше 12 років найкращим варіантом рятувальної терапії є використання комбінації низьких доз інгаляційних кортикостероїдів (ІКС) та формотеролу при потребі, що знижує ризик тяжких загострень порівняно з використанням лише сальбутамолу.",
        metadata={"source": "who_guidelines", "topic": "pulmonology", "type": "protocol"}
    ),
    Document(
        page_content="Протокол надання допомоги при гострому інфаркті міокарда з елевацією сегмента ST: Час від першого медичного контакту до проведення черезшкірного коронарного втручання (ЧКВ) не повинен перевищувати 90 хвилин. У разі неможливості ЧКВ протягом 120 хвилин показана тромболітична терапія.",
        metadata={"source": "moh_ukraine_protocols", "topic": "cardiology", "type": "protocol"}
    ),
    Document(
        page_content="Ерадикація Helicobacter pylori (Маастрихтський консенсус VI): Стандартом терапії першої лінії є квадротерапія з препаратом вісмуту тривалістю 14 днів (Інгібітор протонної помпи + вісмуту субцитрат + тетрациклін + метронідазол). У регіонах з низькою резистентністю до кларитроміцину допускається потрійна терапія.",
        metadata={"source": "european_guidelines", "topic": "gastroenterology", "type": "protocol"}
    ),
    Document(
        page_content="Клінічна настанова щодо ведення пацієнтів з цукровим діабетом 2 типу: Метформін залишається препаратом першої лінії. При наявності серцево-судинних захворювань або хронічної хвороби нирок до схеми лікування обов'язково додаються інгібітори SGLT2 (емпагліфлозин, дапагліфлозин) або агоністи рецепторів GLP-1.",
        metadata={"source": "ada_standards", "topic": "endocrinology", "type": "protocol"}
    ),
    Document(
        page_content="Протокол лікування гострого середнього отиту у дітей: Амоксицилін є антибіотиком першого вибору в дозі 80-90 мг/кг/добу, розділеній на 2 прийоми, протягом 5-7 днів. При алергії на пеніциліни рекомендовано використання цефалоспоринів II-III покоління (цефуроксим, цефподоксим) або макролідів.",
        metadata={"source": "moh_ukraine_protocols", "topic": "pediatrics", "type": "protocol"}
    ),

    # --- Результати досліджень (Research) ---
    Document(
        page_content="Мета-аналіз Кохрейнівської співдружності щодо ефективності вітаміну D: Дослідження, що охопило понад 11 000 учасників, показало, що щоденний або щотижневий прийом вітаміну D знижує ризик гострих інфекцій дихальних шляхів на 12%. Найбільший ефект спостерігався у пацієнтів з початковим вираженим дефіцитом (рівень 25(OH)D < 25 нмоль/л).",
        metadata={"source": "cochrane_reviews", "topic": "immunology", "type": "research"}
    ),
    Document(
        page_content="Дослідження впливу середземноморської дієти (PREDIMED): Дотримання середземноморської дієти, збагаченої оливковою олією першого віджиму або горіхами, знижує частоту основних серцево-судинних подій (інфаркт, інсульт, серцево-судинна смерть) на 30% у пацієнтів з високим ризиком порівняно з дієтою зі зниженим вмістом жирів.",
        metadata={"source": "clinical_trials_journal", "topic": "nutrition", "type": "research"}
    ),
    Document(
        page_content="Рандомізоване клінічне дослідження ефективності антидепресантів (STAR*D): Згідно з результатами, лише близько 33% пацієнтів досягають повної ремісії після першого курсу лікування селективними інгібіторами зворотного захоплення серотоніну (СІЗЗС). При відсутності ефекту протягом 6-8 тижнів показана зміна препарату або аугментація.",
        metadata={"source": "psychiatry_research", "topic": "psychiatry", "type": "research"}
    ),
    Document(
        page_content="Дослідження ефективності ібупрофену та парацетамолу в педіатрії: Клінічні випробування демонструють, що ібупрофен (10 мг/кг) діє швидше і забезпечує довший жарознижуючий ефект порівняно з парацетамолом (15 мг/кг) у дітей віком від 6 місяців до 12 років. Одночасне застосування обох препаратів не рекомендоване.",
        metadata={"source": "pediatrics_journal", "topic": "pediatrics", "type": "research"}
    ),
    Document(
        page_content="Ефективність пробіотиків при антибіотик-асоційованій діареї: Штами Saccharomyces boulardii та Lactobacillus rhamnosus GG довели високу ефективність у профілактиці антибіотик-асоційованої діареї. Зниження відносного ризику розвитку діареї становить близько 50% при одночасному прийомі з антибактеріальною терапією.",
        metadata={"source": "gastroenterology_studies", "topic": "microbiome", "type": "research"}
    ),
    Document(
        page_content="Новітні дослідження мігрені: Використання моноклональних антитіл до CGRP (кальцитонін-ген спорідненого пептиду), таких як еренумаб або фреманезумаб, знижує частоту нападів мігрені на 50% і більше у пацієнтів з хронічною та епізодичною мігренню, що не реагують на традиційну терапію (бета-блокатори, топірамат).",
        metadata={"source": "neurology_research", "topic": "neurology", "type": "research"}
    ),

    # --- Медичні довідники (Reference / Guidelines) ---
    Document(
        page_content="Фармакологічний довідник: Взаємодія ліків. Небезпечна комбінація: Триптани (суматриптан, золмітриптан) для лікування мігрені не повинні застосовуватися одночасно з антидепресантами групи СІЗЗС (флуоксетин, сертралін) через високий ризик розвитку серотонінового синдрому (гіпертермія, тремор, порушення свідомості).",
        metadata={"source": "medical_reference_book", "topic": "pharmacology", "type": "reference"}
    ),
    Document(
        page_content="Токсикологічний довідник: Отруєння парацетамолом. Гепатотоксична доза парацетамолу для дорослих становить понад 10-15 г одноразово, або понад 150 мг/кг. Специфічним антидотом є N-ацетилцистеїн. Його введення найбільш ефективне протягом перших 8 годин після передозування.",
        metadata={"source": "toxicology_manual", "topic": "toxicology", "type": "reference"}
    ),
    Document(
        page_content="Довідник лабораторних показників: Критерії анемії за даними ВООЗ. Діагноз анемії встановлюється при зниженні рівня гемоглобіну <130 г/л у чоловіків, <120 г/л у невагітних жінок та <110 г/л у вагітних жінок. Мікроцитарна анемія (MCV < 80 фл) найчастіше свідчить про дефіцит заліза.",
        metadata={"source": "lab_reference", "topic": "hematology", "type": "reference"}
    ),
    Document(
        page_content="Довідник класифікації хвороб нирок: Швидкість клубочкової фільтрації (ШКФ) розраховується за формулою CKD-EPI. Хронічна хвороба нирок 3-ї стадії (ХХН G3) діагностується при стабільному зниженні ШКФ до 30-59 мл/хв/1.73 м2 протягом щонайменше 3 місяців.",
        metadata={"source": "nephrology_reference", "topic": "nephrology", "type": "reference"}
    ),
    Document(
        page_content="Довідник симптомів червоних прапорців: Головний біль. Симптоми, що вимагають негайної нейровізуалізації (МРТ/КТ): раптовий 'громоподібний' головний біль (підозра на субарахноїдальний крововилив), головний біль, що супроводжується вогнищевою неврологічною симптоматикою, або новий головний біль у пацієнтів старше 50 років.",
        metadata={"source": "medical_reference_book", "topic": "neurology", "type": "reference"}
    ),
    Document(
        page_content="Довідник з вакцинації: Планова вакцинація проти вірусу папіломи людини (ВПЛ). Вакцинація рекомендована для дівчат та хлопчиків у віці 9-14 років до початку статевого життя (дводозова схема з інтервалом 6-12 місяців). Вона забезпечує понад 90% захисту від раку шийки матки та інших ВПЛ-асоційованих онкологічних захворювань.",
        metadata={"source": "immunization_schedule", "topic": "infectious_diseases", "type": "reference"}
    ),
    Document(
        page_content="Ендокринологічний довідник: Діагностика гіпотиреозу. Первинний гіпотиреоз підтверджується при підвищеному рівні тиреотропного гормону (ТТГ > 4.0 мМО/л) та зниженому рівні вільного тироксину (вТ4). Субклінічний гіпотиреоз визначається при підвищеному ТТГ на тлі нормального рівня вТ4.",
        metadata={"source": "medical_reference_book", "topic": "endocrinology", "type": "reference"}
    )
]


########################################### Векторне сховище ####################################################
# embeddings = OpenAIEmbeddings(
#     model="openai/text-embedding-3-small",
#     openai_api_base="https://openrouter.ai/api/v1",
#     openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
# )

# vectorstore = Chroma(
#     collection_name="course_knowledge",
#     embedding_function=embeddings
# )

# # Індексуємо лише один раз: інакше при кожному запуску скрипта база
# # наповнюється дублікатами тих самих документів.
# if vectorstore._collection.count() == 0:
#     vectorstore.add_documents(MOCK_DOCUMENTS)

# retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
_vectorstore = None
_retriever = None

def get_retriever():
    global _vectorstore, _retriever
    
    # Якщо вже створено, просто повертаємо
    if _retriever is not None:
        return _retriever
        
    print("Initializing ChromaDB...", file=sys.stderr) # Логування безпечне, якщо в stderr
    
    embeddings = OpenAIEmbeddings(
        model="openai/text-embedding-3-small",
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
    )

    _vectorstore = Chroma(
        collection_name="course_knowledge",
        embedding_function=embeddings
    )

    if _vectorstore._collection.count() == 0:
        _vectorstore.add_documents(MOCK_DOCUMENTS)

    _retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    return _retriever

########################################### Pydantic схеми ####################################################
class SymptomLookupInput(BaseModel):
    """Вхідні данні для запита пошуку інформації про симптом"""
    model_config = ConfigDict(str_strip_whitespace=True,
                              str_to_lower=True)
    
    symptom: str = Field(pattern=r"^[а-щА-ЩЬьЮюЯяЇїІіЄєҐґ ]*$",
                         description="Опис або назва симптому",
                         min_length=2)


class DrugInfoInput(BaseModel):
    """Вхідні данні для запита пошуку інформації про медичний препарат"""
    model_config = ConfigDict(str_strip_whitespace=True,
                              str_to_lower=True)
    
    drug_name: str = Field(pattern=r"^[а-щА-ЩЬьЮюЯяЇїІіЄєҐґ ]*$",
                           description="Назва препарата",
                           min_length=2)


class BmiCalculatorInput(BaseModel):
    """Вхідні данні для запита розрахунку індекса маси тіла(BMI)"""

    weight_kg: float = Field(description="Вага людини в кг", gt=0)
    height_cm: float = Field(description="Зріст людини в см", gt=0)


class CalorieCalculatorInput(BaseModel):
    """Вхідні данні для запита розрахунку калорій"""
    model_config = ConfigDict(str_strip_whitespace=True,
                              str_to_lower=True)
    
    age: int = Field(description="Вік людини", ge=0)
    weight: float = Field(description="Вага людини в кг", gt=0)
    height: float = Field(description="Зріст людини в см", gt=0)
    activity_level: Literal["sedentary", "light", "moderate", "active",
                            "very_active"] = Field(description="Рівень активності людини")
    gender: Literal["male", "female"] = Field(description="Стать людини")

class RAGInput(BaseModel):
    """Вхідні дані для пошуку в базі знань курсу."""
    model_config = ConfigDict(str_strip_whitespace=True,
                              str_to_lower=True)
    
    query: str = Field(description="Пошуковий запит до бази знань", 
                       min_length=3)


class DosageRecommendationInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, str_to_lower=True)
    drug: str = Field(
        pattern=r"^[а-щА-ЩЬьЮюЯяЇїІіЄєҐґa-zA-Z \-]*$", 
        description="Назва медичного препарату.",
        min_length=2
    )
    weight_kg: float = Field(description="Вага пацієнта в кілограмах.", gt=0)
    age: int = Field(description="Повний вік пацієнта в роках.", ge=0)

########################################### Tools ####################################################
@mcp.tool()
def symptom_lookup(symptom: str) -> str:
    """Шукає інформацію про симптом у базі даних."""

    try:
        validated_data = SymptomLookupInput(symptom=symptom)
        clean_symptom = validated_data.symptom
    except ValueError as e:
        return f"Помилка формату: {e}. Використовуйте лише українські літери."

    return MOCK_SYMPTOMS.get(
        clean_symptom,
        f"Інформацію про симптом '{clean_symptom}' не знайдено. Зверніться до лікаря для точної діагностики."
    )


@mcp.tool()
def drug_info(drug_name: str) -> str:
    """Шукає довідкову інформацію про медичний препарат."""

    try:
        validated_data = DrugInfoInput(drug_name=drug_name)
        clean_name = validated_data.drug_name
    except ValueError as e:
        return f"Помилка формату: {e}. Використовуйте лише українські літери."

    return MOCK_DRUGS.get(
        clean_name,
        f"Препарат '{clean_name}' не знайдено у базі. Проконсультуйтеся з фармацевтом або лікарем."
    )


@mcp.tool()
def bmi_calculator(weight_kg: float, height_cm: float) -> Dict[str, Union[float, str]]:
    """Розраховує індекс маси тіла (BMI) та визначає категорію."""
    if height_cm <= 0 or weight_kg <= 0:
        return {"error": "Вага та зріст мають бути більшими за нуль."}

    try:
        validated_data = BmiCalculatorInput(
            weight_kg=weight_kg, height_cm=height_cm)
    except ValueError as e:
        return {"error": f"Помилка формату: {e}."}

    height_m = height_cm / 100
    bmi = round(weight_kg / (height_m ** 2), 1)

    if bmi < 18.5:
        category = "Недостатня маса тіла"
    elif 18.5 <= bmi <= 24.9:
        category = "Нормальна маса тіла"
    elif 25.0 <= bmi <= 29.9:
        category = "Надлишкова маса тіла (передожиріння)"
    else:
        category = "Ожиріння"

    return {"bmi": bmi, "category": category}


@mcp.tool()
def calorie_calculator(age: int, weight: float, height: float, activity_level: str, gender: str = "male") -> Dict[str, Union[float, str]]:
    """
    Розраховує добову норму калорій за формулою Міффліна-Сан Жеора.
    activity_level: 'sedentary', 'light', 'moderate', 'active', 'very_active'
    """

    try:
        validated_data = CalorieCalculatorInput(
            age=age, weight=weight, height=height, activity_level=activity_level, gender=gender)
    except ValueError as e:
        return {"error": f"Помилка формату: {e}."}

    activity_multipliers = {
        'sedentary': 1.2,       # Сидячий спосіб життя
        'light': 1.375,         # Легка активність (1-3 дні на тиждень)
        'moderate': 1.55,       # Помірна активність (3-5 днів на тиждень)
        'active': 1.725,        # Висока активність (6-7 днів на тиждень)
        'very_active': 1.9
    }

    multiplier = activity_multipliers.get(activity_level.lower(), 1.2)

    # Базовий рівень метаболізму (BMR)
    if gender.lower() == 'male':
        bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        bmr = (10 * weight) + (6.25 * height) - (5 * age) - 161

    daily_calories = round(bmr * multiplier)

    return {
        "bmr": round(bmr),
        "daily_calorie_target": daily_calories,
        "note": "Це орієнтовний розрахунок. Для точної дієти потрібна консультація дієтолога."
    }

@mcp.tool()
async def knowledge_search(query: str) -> str:
    """Шукає релевантну інформацію в базі знань.
    """
    logging.info(f"Початок пошуку для запиту: {query}")
    try:
        validated_data = RAGInput(query=query)
        clean_query = validated_data.query
    except ValueError as e:
        logging.error(f"Помилка валідації: {e}")
        return f"Помилка формату: {e}."

    try:
        logging.info("Ініціалізація ChromaDB / OpenAIEmbeddings...")
        retriever = get_retriever()
        
        #logging.info("Виклик retriever.ainvoke()...")

        #docs = await retriever.ainvoke(clean_query) 
        
        # Витягуємо саму векторну базу з обгортки retriever
        v_store = retriever.vectorstore
        
        logging.info("Отримуємо вектор запиту асинхронно...")
        # 1. Асинхронний запит до OpenRouter (не блокує систему)
        query_vector = await v_store.embeddings.aembed_query(clean_query)
        
        logging.info("Локальний пошук у ChromaDB...")
        # 2. Синхронний пошук по вектору (уникає конфлікту SQLite)
        docs = v_store.similarity_search_by_vector(query_vector, k=3)

        
        logging.info(f"Знайдено документів: {len(docs)}")
        
        if not docs:
            return json.dumps({"status": "not_found", "query": clean_query})
            
        results = []
        for d in docs:
            results.append({
                "content": d.page_content,
                "source": d.metadata.get("source", "?"),
                "topic": d.metadata.get("topic", "?"),
            })
        return json.dumps({"status": "ok", "query": clean_query, "results": results}, ensure_ascii=False)
        
    except Exception as e:
        logging.error(f"КРИТИЧНА ПОМИЛКА: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool()
def dosage_recommendation(drug: str, weight_kg: float, age: int) -> str:
    """Розраховує чорнове дозування для затвердження лікарем."""
    
    try:
        validated_data = DosageRecommendationInput(drug=drug, weight_kg=weight_kg, age=age)
        clean_drug = validated_data.drug
    except ValueError as e:
        return f"Помилка формату: {e}."
    
    if clean_drug in MOCK_DOSAGES:
        multiplier = MOCK_DOSAGES[clean_drug]
        dose = round(multiplier * weight_kg, 2)
        calculated_dose = f"{dose} мг (розрахунок: {multiplier} мг/кг)"
    else:
        calculated_dose = "Немає даних у базі (потрібен ручний розрахунок лікарем)"

    return (
        f"ЧОРНОВИЙ РОЗРАХУНОК (ОЧІКУЄ ПІДТВЕРДЖЕННЯ):\n"
        f"Препарат: {drug.capitalize()}\n"
        f"Параметри: {age} років, {weight_kg} кг\n"
        f"Пропонована доза: {calculated_dose}"
    )

if __name__ == '__main__':
    mcp.run(transport='stdio')
