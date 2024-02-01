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
	#Extracted from hwchase17/structured-chat-agent, 
	prompt.messages[0].prompt.template = """
		'Respond to the human as helpfully and accurately as possible.
		 When using the booking tool (if available), return the list of the best apartments according to the users input in response. The response should contain all the data about the apartments you have, including the link where the user can look at an apartment.
		 You have access to the following tools:\n\n{tools}\n\nUse a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).\n\nValid "action" values: "Final Answer" or {tool_names}\n\nProvide only ONE action per $JSON_BLOB, as shown:\n\n```\n{{\n  "action": $TOOL_NAME,\n  "action_input": $INPUT\n}}\n```\n\nFollow this format:\n\nQuestion: input question to answer\nThought: consider previous and subsequent steps\nAction:\n```\n$JSON_BLOB\n```\nObservation: action result\n... (repeat Thought/Action/Observation N times)\nThought: I know what to respond\nAction:\n```\n{{\n  "action": "Final Answer",\n  "action_input": "Final response to human"\n}}\n\nBegin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'
	"""
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