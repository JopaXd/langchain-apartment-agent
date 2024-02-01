from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Optional, List
import g4f

class G4FLLM(LLM):
	gpt_model:str = "gpt-4"

	@property
	def _providers(self) -> list:
		if self.gpt_model == "gpt-4":
			return [
				g4f.Provider.Bing,
				g4f.Provider.GeekGpt,
				g4f.Provider.GptChatly,
				g4f.Provider.Liaobots
			]
		elif self.gpt_model == "gpt-3.5":
			return [
				g4f.Provider.GeekGpt,
				g4f.Provider.GptChatly,
				g4f.Provider.Liaobots,
				g4f.Provider.Yqcloud,
				g4f.Provider.ChatBase,
				g4f.Provider.ChatgptAi,
				g4f.Provider.FakeGpt,
				g4f.Provider.GPTalk,
				g4f.Provider.GptForLove,
				g4f.Provider.GptGo,
				g4f.Provider.Hashnode,
				g4f.Provider.You,
				g4f.Provider.ChatForAi,
				g4f.Provider.AItianhuSpace
			]
		else:
			return []

	@property
	def _llm_type(self) -> str:
		return "custom"

	def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
		for provider in self._providers:
			try:
				if self.gpt_model == "gpt-4":
					model = g4f.models.gpt_4
				elif self.gpt_model == "gpt-3.5":
					model = g4f.models.gpt_35_turbo
				out = g4f.ChatCompletion.create(
					model=model,
					messages=[{"role": "user", "content": prompt}],
					provider = provider,
				)
				if stop:
					stop_indexes = (out.find(s) for s in stop if s in out)
					min_stop = min(stop_indexes, default=-1)
					if min_stop > -1:
						out = out[:min_stop]
				return out
			except Exception as e:
				print(e)
				continue
			break
		return ""

	async def _acall(self, prompt:str, run_manager:Optional[AsyncCallbackManagerForLLMRun], stop: Optional[List[str]] = None, **kwargs):
		for provider in self._providers:
			try:
				if self.gpt_model == "gpt-4":
					model = g4f.models.gpt_4
				elif self.gpt_model == "gpt-3.5":
					model = g4f.models.gpt_35_turbo
				out = await g4f.ChatCompletion.create_async(
					model=model,
					messages=[{"role": "user", "content": prompt}],
					provider = provider,
				)
				if stop:
					stop_indexes = (out.find(s) for s in stop if s in out)
					min_stop = min(stop_indexes, default=-1)
					if min_stop > -1:
						out = out[:min_stop]
				return out
			except Exception as e:
				print(e)
				continue
			break
		return ""