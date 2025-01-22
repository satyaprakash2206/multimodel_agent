from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import openai
import os
from dotenv import load_dotenv
openai.api_key = os.getenv("OPENAI_API_KEY")

web_search_agent = Agent(
    name='web_search_agent',
    role='Search the web for the information',
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["ALways provide the source"],
    show_tool_calls=True,
    markdown=True
)

financial_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the information"],
    show_tool_calls=True,
    markdown=True
)


multi_ai_agent = Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_search_agent, financial_agent],
    instructions=["always include the source","use tables to display the information"],
    show_tool_calls=True,
    markdown=True
)

multi_ai_agent.print_response("summarize analyst recommendations and news for NVDA",stream=True)