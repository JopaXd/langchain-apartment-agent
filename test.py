from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from kor import create_extraction_chain, Object, Text, Bool
from g4fllm import G4FLLM
import pprint

schema = Object(
		id="apartments",
		description = ("Apartments that the user might be interested in according to their input."),
		attributes=[
			Text(id="apartment_name", description="Name of apartment"),
			Text(id="apartment_link", description="Link of the page where the user can look at the apartment", examples=[("https://booking.com/hotel/bg...", "Example of the link of an appartment")]),
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

# schema = {
# 	"properties" : {
# 		"apartment_name" : {"type" : "string"},
# 		"apartment_link": {"type" : "string"},
# 		"apartment_location": {"type" : "string"},
# 		"apartment_distance_from_center": {"type": "string"},
# 		"apartment_distance_from_beach": {"type" : "string"},
# 		"apartment_free_cancellation": {"type" : "boolean"},
# 		"apartment_taxes": {"type": "string"},
# 		"apartment_description": {"type": "string"},
# 		"apartment_price": {"type": "string"}
# 	},
# 	"required" : ["apartment_name", "apartment_link", "apartment_location", "apartment_distance_from_center", "apartment_distance_from_beach", "apartment_free_cancellation", "apartment_taxes", "apartment_description", "apartment_price"]
# }

llm = G4FLLM()

def extract(content:str, schema):
	print(content)
	return create_extraction_chain(llm, schema, encoder_or_encoder_class='json').invoke(content)

def scrape_booking(urls, schema):
	loader = AsyncHtmlLoader(urls)
	docs = loader.load()
	for doc in docs:
		doc.page_content = doc.page_content.replace(u"\xa0", u"").strip()
	bs_transformer = BeautifulSoupTransformer()
	docs_transformed = bs_transformer.transform_documents(docs, tags_to_extract=["div", 'span', 'p', 'li', 'a', 'strong'])
	print("Extracting content with LLM")
	splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=2048, chunk_overlap=0)
	splits = splitter.split_documents(docs_transformed)
	extracted_content = extract(splits[0].page_content, schema)
	pprint.pprint(extracted_content)
	return extracted_content

urls = ["https://www.booking.com/searchresults.html?ss=Pomorie%2C+Burgas+Province%2C+Bulgaria&lang=en-us&checkin=2024-07-27&checkout=2024-08-05&group_adults=2&no_rooms=1&group_children=0&nflt=price%3DEUR-min-40-1&selected_currency=EUR"]

extracted_content = scrape_booking(urls, schema=schema)