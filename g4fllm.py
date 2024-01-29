from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
import g4f

class G4FLLM(LLM):

	@property
	def _providers(self) -> list:
		return [
			g4f.Provider.Bing,
			g4f.Provider.GeekGpt,
			g4f.Provider.GptChatly,
			g4f.Provider.Liaobots
		]

	@property
	def _llm_type(self) -> str:
		return "custom"

	def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs) -> str:
		for provider in self._providers:
			try:
				out = g4f.ChatCompletion.create(
					model=g4f.models.gpt_4,
					messages=[{"role": "user", "content": prompt}],
					provider = provider,
				)
			except Exception as e:
				print(e)
				continue
			break
		if stop:
			stop_indexes = (out.find(s) for s in stop if s in out)
			min_stop = min(stop_indexes, default=-1)
			if min_stop > -1:
				out = out[:min_stop]
		return out

	async def _acall(self, prompt:str, run_manager:Optional[AsyncCallbackManagerForLLMRun], stop: Optional[List[str]] = None, **kwargs):
		for provider in self._providers:
			try:
				out = await g4f.ChatCompletion.create_async(
					model=g4f.models.gpt_4,
					messages=[{"role": "user", "content": prompt}],
					provider = provider,
				)
			except Exception as e:
				print(e)
				continue
			break
		if stop:
			stop_indexes = (out.find(s) for s in stop if s in out)
			min_stop = min(stop_indexes, default=-1)
			if min_stop > -1:
				out = out[:min_stop]
		return out