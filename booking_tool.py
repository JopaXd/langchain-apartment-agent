from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.pydantic_v1 import BaseModel, Field
from langchain.callbacks.manager import (
	CallbackManagerForToolRun,
)
from langchain.tools import BaseTool
from kor import create_extraction_chain, Object, Text, Bool
from typing import Optional, Type
from g4fllm import G4FLLM
import pprint
import json
import re

class BookingInput(BaseModel):
	location: str = Field(description="Location where the user wants to travel/location of the apartment.")
	checkInDate: str = Field(description="Date of checking in. Example: 2024-08-05")
	checkOutDate: str = Field(description="Date of checking out. Example: 2024-08-05")
	numberOfAdults: int =  Field(description="Number of adults that will be in the apartment.")
	numberOfChildren: int =  Field(description="Number of children that will be in the apartment.")
	numberOfRooms: int = Field(description="Number of rooms in the apartment")
	#Due to the possibility of "min" and "max" values, these should be string.
	minimumPricePerNight: str = Field(description="The minimum that the user is willing to spend per night (in euros). The lower value of the price range.")
	maximumPricePerNight: str = Field(description="The maximum that the user is willing to spend per night (in euros). The higher value of the price range.")

class BookingTool(BaseTool):
	name = "Booking"
	description = (
		"Use this tool when you need to find, and recommend the best apartments according to the user input."
		"Given the check in and check out date, number of rooms in the apartment, number of adults and children in the apartment, and minimum and maximum price per night"
		"To use the tool you must provide all of the following parameters "
		"['location', 'checkInDate', 'checkOutDate', 'numberOfAdults', 'numberOfRooms', 'numberOfChildren', 'minimumPricePerNight', 'maximumPricePerNight']."
	)
	args_schema: Type[BaseModel] = BookingInput
	# What this does is it makes the ai only return the result of the tool.
	# In this case a json of apartments.
	# We don't need that. We need the ai to return its won response based on the tool's result.
	# return_direct: bool = True

	def _extract(self, content:str, schema):
		llm = G4FLLM()
		return create_extraction_chain(llm, schema, encoder_or_encoder_class='json').invoke(content)

	# Sometimes, the response is not returned fully according to the schema, thus producing errors.
	# Apparently, better description can help mitigate this.
	def _scrape_booking(self, url):
		schema = Object(
			id="apartments",
			description = ("Apartments that the user might be interested in according to their input."),
			attributes=[
				Text(id="apartment_name", description="Name of apartment"),
				# Text(id="apartment_link", description="Link of the page where the user can look at the apartment", examples=[("https://booking.com/hotel/bg...", "Example of the link of an appartment")]),
				Text(id="apartment_location", description="Location of the apartment"),
				Text(id="apartment_distance_from_center", description="Distance of the apartment from the city center"),
				Text(id="apartment_distance_from_beach", description="Distance of the apartment from the beach"),
				Bool(id="apartment_free_cancellation", description="Can the apartment reservation be cancelled for free"),
				Text(id="apartment_taxes", description="Taxes for apartment"),
				Text(id="apartment_description", description="Description of apartment, for example, rooms, etc..."),
				Text(id="apartment_price", description="Price of the apartment, the lower price that is present."),
				Text(id="apartment_rating", description="Rating of the apartment")
			],
			many=True
		)
		loader = AsyncHtmlLoader([url])
		docs = loader.load()
		for doc in docs:
			doc.page_content = doc.page_content.replace(u"\xa0", u"").strip()
		bs_transformer = BeautifulSoupTransformer()
		docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["div", 'span', 'p', 'li', 'a', 'strong'])
		print("Extracting content with LLM")
		splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2048, chunk_overlap=0)
		splits = splitter.split_documents(docs_transformed)
		extracted_content = self._extract(splits[0].page_content, schema)
		pprint.pprint(extracted_content)
		# print(extracted_content["text"]["data"]["apartments"])
		wrapper = DuckDuckGoSearchAPIWrapper(region="wt-wt", time=None, max_results=3)
		search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="text")
		try:
			for apartment in extracted_content["text"]["data"]["apartments"]:
				apartment_url = re.search("(?P<url>https?://[^\s]+)", search.run(f"{apartment['apartment_name']} {apartment['apartment_location']} booking")).group("url")[:-2]
				if apartment_url:
					apartment["apartment_url"] = apartment_url
			return extracted_content["text"]["data"]["apartments"]
		except KeyError:
			#Parsing of scraper output failed
			#Working with raw output.
			print("KeyErorr occured, using raw json.")
			content = extracted_content["text"]["raw"].replace("<json>\n", "").replace("</json>", "")
			content_json = json.loads(content)
			for apartment in content_json["apartments"]:
				apartment_url = re.search("(?P<url>https?://[^\s]+)", search.run(f"{apartment['apartment_name']} {apartment['apartment_location']} booking")).group("url")[:-2]
				if apartment_url:
					apartment["apartment_url"] = apartment_url
			return content_json
		
	def _run(
		self,
		location:str,
		checkInDate:str,
		checkOutDate:str,
		numberOfAdults:int,
		numberOfChildren:int,
		numberOfRooms:int,
		minimumPricePerNight:str="min",
		maximumPricePerNight:str="max",
		run_manager: Optional[CallbackManagerForToolRun] = None
		) -> str:
		"""Use the tool."""
		return str(self._scrape_booking(f"https://www.booking.com/searchresults.html?ss={location}&lang=en-us&checkin={checkInDate}&checkout={checkOutDate}&group_adults={numberOfAdults}&no_rooms={numberOfRooms}&group_children={numberOfChildren}&nflt=price%3DEUR-{minimumPricePerNight}-{maximumPricePerNight}-1&selected_currency=EUR"))

	def _arun(
		self,
		location:str,
	 	checkInDate:str,
	 	checkOutDate:str,
	 	numberOfAdults:int,
	  	numberOfChildren:int,
	  	numberOfRooms:int,
	  	minimumPricePerNight:str="min",
	  	maximumPricePerNight:str="max",
	  	run_manager: Optional[CallbackManagerForToolRun] = None) -> str:
		"""Use the tool asynchronously."""
		raise NotImplementedError("Calculator does not support async")