from langchain.agents import create_react_agent, Tool, AgentExecutor
from bs4 import BeautifulSoup
from langchain import hub
from g4fllm import G4FLLM
import requests
import os

headers={"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"}

resp = requests.get("https://www.booking.com/searchresults.html?ss=Pomorie%2C+Burgas+Province%2C+Bulgaria&lang=en-us&checkin=2024-07-27&checkout=2024-08-05&group_adults=2&no_rooms=1&group_children=0&nflt=price%3DEUR-min-40-1&selected_currency=EUR", headers=headers)
soup = BeautifulSoup(resp.text, "lxml")

apartments = soup.find_all("div", class_=["c82435a4b8", "a178069f51", "a6ae3c2b40", "a18aeea94d", "d794b7a0f7", "f53e278e95", "c6710787a4"])

data = {"apartments": []}

for ap in apartments:
	apartment_data = {}
	try:
		#Grab title (and link)
		link = ap.find("a", class_="a78ca197d0")
		title = link.find("div").text
		print(link["href"])
		print(title)
		apartment_data["name"] = title
		apartment_data["link"] = link["href"]
		#GET DESCRIPTION
		ap_res = requests.get(link["href"], headers=headers)
		ap_soup = BeautifulSoup(ap_res.text, "lxml")
		description = ap_soup.find("p", class_=["a53cbfa6de", "b3efd73f69"]).text
		print(description)
		apartment_data["description"] = description
		#GET RATING
		rating = ap.find("div", class_=["a3b8729ab1", "d86cee9b25"]).text
		print(rating)
		apartment_data["rating"] = float(rating)
		#PRICE
		price = ap.find("span", class_=["f6431b446c", "fbfd7c1165", "e84eb96b1f"]).text.replace(u'\xa0', u"")
		print(price)
		apartment_data["price"] = price
		#TAXES?
		taxes = ap.find("div", attrs={'data-testid': 'taxes-and-charges'}).text
		if taxes:
			print(taxes)
		apartment_data["taxes"] = taxes
		#Distances:
		distance_info = ap.find_all("span", class_="aee5343fdb")
		apartment_data["distance_data"] = []
		#Remove location and show on map
		for distance in distance_info[2:]:
			apartment_data["distance_data"].append(distance.text)
		#Other INFO:
		apartment_data["other_info"] = []
		other_info = ap.find("div", class_="c59cd18527")
		for info in other_info.ul.find_all("li"):
			apartment_data["other_info"].append(info.text)
		print("*************************")
		data["apartments"].append(apartment_data)
	except Exception as e:
		print(e)

print(data)