from langchain.agents import AgentExecutor, create_structured_chat_agent
from booking_tool import BookingTool
from langchain import hub
from g4fllm import G4FLLM
import chainlit as cl

llm = G4FLLM()

@cl.on_chat_start
def start():
	tools = [BookingTool()]
	prompt = hub.pull("hwchase17/structured-chat-agent")
	llm = G4FLLM(gpt_model="gpt-4")
	agent = create_structured_chat_agent(llm, tools, prompt)
	agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)
	cl.user_session.set("agent", agent_executor)

@cl.on_message
async def main(message):
	agent = cl.user_session.get("agent")
	cb = cl.LangchainCallbackHandler(stream_final_answer=True)
	agent.callbacks = [cb]
	await cl.make_async(agent.invoke)({"input": message.content}, callbacks=[cb])