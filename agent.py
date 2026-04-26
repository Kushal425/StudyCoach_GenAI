import os
from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from rag_setup import RAGManager
from dotenv import load_dotenv

load_dotenv()

# Define the state for the LangGraph
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    student_profile: str
    pass_probability: float

# Initialize RAG
rag_manager = RAGManager()
rag_manager.initialize_db()

@tool
def search_educational_content(query: str) -> str:
    """Searches the educational database for tutorials and explanations related to the query."""
    results = rag_manager.search(query)
    if not results:
        return "No specific educational content found for this query."
    
    formatted_results = "\\n".join([f"- {r['content']} (Source: {r['source']})" for r in results])
    return f"Here are the relevant study materials found:\\n{formatted_results}"

tools = [search_educational_content]

def initialize_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        # Fallback to a dummy model or raise a clear error
        print("WARNING: GROQ_API_KEY not found. Please add it to .env")
        # For the sake of this prototype, if no key, we might crash on execution,
        # but we initialize ChatGroq anyway to allow user to drop it in.
        return ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")
    return ChatGroq(temperature=0.7, model_name="llama-3.3-70b-versatile")

llm = initialize_llm()
llm_with_tools = llm.bind_tools(tools)

def chat_node(state: AgentState):
    """The main LLM node that processes the conversation."""
    messages = state["messages"]
    
    # Inject student context into the system message if this is the first turn
    if not any(isinstance(m, SystemMessage) for m in messages):
        profile = state.get("student_profile", "Unknown")
        prob = state.get("pass_probability", 0.0)
        
        sys_msg = SystemMessage(
            content=f"You are an AI Study Coach. You are interacting with a student whose learning profile is '{profile}' "
                    f"and their current estimated pass probability is {prob:.1%}. "
                    f"Tailor your coaching style accordingly. For 'At-Risk' students, be encouraging and break concepts down. "
                    f"For 'High-Performer' students, challenge them with advanced concepts."
        )
        messages = [sys_msg] + messages
        
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: AgentState):
    """Executes the tool calls made by the LLM."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if not last_message.tool_calls:
        return {"messages": []}
        
    tool_messages = []
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "search_educational_content":
            result = search_educational_content.invoke(tool_call["args"])
            tool_messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            
    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    """Determines whether to execute a tool or end the turn."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node("agent", chat_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# Compile the graph
agent_executor = workflow.compile()

def run_agent(user_input: str, student_profile: str, pass_probability: float, memory: list = None):
    """Helper function to run the agent from the UI."""
    if memory is None:
        memory = []
        
    state = {
        "messages": memory + [HumanMessage(content=user_input)],
        "student_profile": student_profile,
        "pass_probability": pass_probability
    }
    
    result = agent_executor.invoke(state)
    return result["messages"]
