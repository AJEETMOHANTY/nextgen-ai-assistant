from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Define the graph
graph = StateGraph(ChatState)

# add nodes
graph.add_node("chat", chat_node)

# add edges
graph.add_edge(START, "chat")
graph.add_edge("chat", END)

# add checkpoint
checkpointer = InMemorySaver()

# compile the graph
chatbot = graph.compile(checkpointer=checkpointer)