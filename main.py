from langchain.agents import AgentExecutor, create_structured_chat_agent
from booking_tool import BookingTool
from langchain import hub
from g4fllm import G4FLLM

llm = G4FLLM()

if __name__ == "__main__":
	tools = [BookingTool()]
	prompt = hub.pull("hwchase17/structured-chat-agent")
	llm = G4FLLM(gpt_model="gpt-4")
	agent = create_structured_chat_agent(llm, tools, prompt)
	agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, max_iterations=10)
	agent_executor.invoke({"input": "I am looking for an apartment in Pomorie for two with one room from 26th of July, to 5th of august this year. The price for one night should be between 10 and 50 CAD."})