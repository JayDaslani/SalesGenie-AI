from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing import TypedDict, List, Annotated
import operator
import os
from dotenv import load_dotenv

from tools import analyze_sales, get_forcast, get_customer_segments, get_quick_stats

load_dotenv()

llm = ChatGroq(
    model='openai/gpt-oss-20b', 
    api_key=os.getenv('GROQ_API_KEY'),
    temperature=0.3
)

tools = [
    analyze_sales,
    get_forcast,
    get_customer_segments,
    get_quick_stats
]

tools_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

class AgentState(TypedDict):
    messages: Annotated[List, operator.add]

def llm_node(state: AgentState):
    system = SystemMessage(content="""
You are SalesGenie AI - an Intelligent 
sales analytics assistant.

You help users understand their sales data.
        
Available tools:
- analyze_sales: Sales analysis
- get_forcast: Future Predictions
- get_customer_segments: Customer info
- get_quick_stats: Quick metrics
                           
Always use tools to get data.
Then explain insights clearly.
Be concise and business-focused.
""")
    
    messages = [system] + state['messages']
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    last_message = state['messages'][-1]
    results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call['name']
        tool_args = tool_call['args']

        if tool_name in tools_map:
            result = tools_map[tool_name].invoke(tool_args)
        else:
            result = f"Tool not found : {tool_name}"

        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call['id'])
        )
    return {'messages': results}

def should_use_tool(state: AgentState):
    last = state['messages'][-1]
    if (hasattr(last, 'tool_calls') and last.tool_calls):
        return "tool"
    return "end"

memory = MemorySaver()

graph = StateGraph(AgentState)
graph.add_node('llm', llm_node)
graph.add_node('tool', tool_node)

graph.set_entry_point('llm')

graph.add_conditional_edges(
    'llm',
    should_use_tool,
    {'tool': 'tool', 'end': END}
)    

graph.add_edge('tool', 'llm')

agent_app = graph.compile(checkpointer=memory)

def chat(message: str, session_id: str='default') -> str:
    config = {
        "configurable": {
            'thread_id': session_id
        }
    }

    result = agent_app.invoke(
        {
            'messages': [
                HumanMessage(content=message)
            ]
        },
        config=config
    )

    for msg in reversed(result['messages']):
        if (isinstance(msg, AIMessage) and msg.content):
            return msg.content

    return "No response generated"

print("=== SalesGenie AI Agent ===")

tests = [
    'What is the total revenue ?',
    'Which is the best-performing region?',
    'Give me the forecast for the next 3 months.',
    'Explain the Customer Segments.',
    'Which Category is the best ?'
]

for q in tests:
    print(f"Q : {q}")
    response = chat(q)
    print(f"A: {response[:200]}\n")
    print('-'*40)

