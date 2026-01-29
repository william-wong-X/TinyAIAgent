from typing import TypedDict, Annotated, Generator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from config.config import AppConfig
from app.model_client import create_llm
from app.tools.registry import create_tools

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

class Agent:
    def __init__(self, config: AppConfig):
        self.config = config
        self.llm = create_llm(config)
        self.tools = create_tools(config)
        self.app = self._build_graph()
    
    def _build_graph(self):
        llm_with_tools = self.llm.bind_tools(self.tools)

        def agent_node(state: AgentState):
            return {"messages": [llm_with_tools.invoke(state["messages"])]}
        
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(self.tools))

        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges("agent", tools_condition)
        workflow.add_edge("tools", "agent")

        checkpointer = MemorySaver()
        return workflow.compile(checkpointer=checkpointer)
    
    def invoke(self, message: str, thread_id: str) -> str:
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=message)]}

        final_state = self.app.invoke(inputs, config=config)
        return final_state["messages"][-1].content
    
    def stream(self, message: str, thread_id: str) -> Generator[str, None, None]:
        config = {"configurable": {"thread_id": thread_id}}
        inputs = {"messages": [HumanMessage(content=message)]}

        event_stream = self.app.stream(inputs, config=config, stream_mode="messages")
        for msg, metadata in event_stream:
            if isinstance(msg, AIMessage) and msg.content:
                yield msg.content