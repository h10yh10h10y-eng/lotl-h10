# taba_agent.py

import requests
from bs4 import BeautifulSoup
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
import os

# 1. כלי לגירוד (Scraping) וניתוח תוצאות חיפוש תב"ע
@tool
def search_taba_info(gush: str = "", chelka: str = "", plan: str = "", locality: str = ""):
    """
    כלי לחיפוש תכניות בניין עיר באתר tabainfo.co.il.
    מקבל פרמטרים כמו גוש, חלקה, מספר תכנית או ישוב ומחזיר את תוצאות החיפוש.
    """
    base_url = 'https://www.tabainfo.co.il/תבע/חיפוש'
    params = {}
    if plan:
        params['number'] = plan
    if locality:
        params['locality'] = locality
    if gush:
        params['block'] = gush
        params['__Invariant'] = 'Block'
    if chelka:
        params['lot'] = chelka
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        plan_items = soup.find_all('div', class_='plan-item')
        if not plan_items:
            plan_items = soup.find_all('a', class_='plan-link')

        for item in plan_items:
            number = item.find('span', class_='plan-number').text if item.find('span', class_='plan-number') else "לא ידוע"
            name = item.find('h4').text if item.find('h4') else "שם לא ידוע"
            results.append(f"{name} ({number})")

        if not results:
            return "לא נמצאו תכניות תואמות בתוצאות האתר. ייתכן והפרמטרים לא נכונים או שהאתר לא מכיל מידע על החיפוש."
            
        return "התוכניות הבאות נמצאו באתר: " + ", ".join(results)
        
    except requests.exceptions.RequestException as e:
        return f"שגיאה בחיבור לאתר: {e}"

# 2. כלי ה-GIS
@tool
def get_parcels_by_zoning(city: str, zoning_type: str):
    """
    כלי לאיתור מגרשים (חלקות) על בסיס עיר וסוג ייעוד (למשל, "תעשייה", "מגורים", "מסחר").
    הכלי מחזיר רשימת מספרי מגרשים ותוכניות תב"ע רלוונטיות.
    """
    if city == "באר שבע" and zoning_type == "אזור תעשייה":
        return "המגרשים 123, 456 נמצאים באזור תעשייה בבאר שבע, תכניות בניין עיר: 100/1, 200/2."
    return "לא נמצאו מגרשים התואמים לקריטריונים."

# 3. כלי ה-TABa (תכנון ובניה)
@tool
def get_plan_details(plan_number: str):
    """
    כלי לקבלת פרטים מפורטים על תכנית בניין עיר (תב"ע) ספציפית.
    הכלי מקבל מספר תכנית בניין עיר ומחזיר סיכום של פרטי התכנית, כולל ייעוד, זכויות בניה ושטחים.
    """
    if plan_number == "100/1":
        return "תכנית 100/1 מיועדת לאזור תעשייה, מאפשרת 500% זכויות בניה, מטרות: הרחבת אזור תעשייה קיים."
    return "לא נמצאו פרטים לתכנית זו."

def create_taba_agent(model_name="gpt-4o-mini", api_key: str = None):
    """
    פונקציה ליצירת והחזרת סוכן AI מומחה בתחום התכנון והבניה.
    הסוכן משלב כלים לחיפוש באתרים חיצוניים ולשליפת מידע מבסיסי נתונים.
    """
    if api_key is None:
        raise ValueError("OpenAI API key must be provided.")
        
    llm = ChatOpenAI(model=model_name, temperature=0, api_key=api_key)
    tools = [search_taba_info, get_parcels_by_zoning, get_plan_details]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        אתה סוכן AI מומחה בתחום תכנון ובניה.
        המטרה שלך היא לענות על שאלות משתמשים בצורה מדויקת ומקצועית באמצעות הכלים שברשותך.
        במקרה הצורך, השתמש בכלי 'search_taba_info' כדי לחפש תוכניות בניין, בכלי 'get_parcels_by_zoning' כדי למצוא חלקות לפי ייעוד, ובכלי 'get_plan_details' כדי לקבל פרטים על תכניות.
        התשובה הסופית צריכה להיות בעברית, ברורה, מפורטת ומבוססת על המידע שהכלים סיפקו.
        אם המידע לא זמין, ציין זאת.
        """
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    
    return agent_executor
