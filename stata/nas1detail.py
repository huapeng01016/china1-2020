import sys
import requests
import json
import argparse
import urllib3
from lxml import html
from random import randint
from time import sleep
from sfi import Frame

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

stockticker = sys.argv[1]
framename = sys.argv[2]

def get_nasdaq_detail(ticker):
	key_stock_dict = {}
	headers = {
		"Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
		"Accept-Encoding":"gzip, deflate",
		"Accept-Language":"en-GB,en;q=0.9,en-US;q=0.8,ml;q=0.7",
		"Connection":"keep-alive",
		"Host":"www.nasdaq.com",
		"Referer":"http://www.nasdaq.com",
		"Upgrade-Insecure-Requests":"1",
		"User-Agent":"Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/64.0.3282.119 Safari/537.36"
	}

	for retries in range(5):
		try:
			url = "http://www.nasdaq.com/symbol/%s"%(ticker)
			response = requests.get(url, headers = headers, verify=False)
		
			if response.status_code!=200:
				raise ValueError("Invalid Response Received From Webserver")

			print("Parsing %s"%(url))
			sleep(randint(1,3))   
			parser = html.fromstring(response.text)
			xpath_head = "//div[@id='qwidget_pageheader']//h1//text()"
			xpath_key_stock_table = '//div[@class="row overview-results relativeP"]//div[contains(@class,"table-table")]/div'
			xpath_open_price = '//b[contains(text(),"Open Price:")]/following-sibling::span/text()'
			xpath_open_date = '//b[contains(text(),"Open Date:")]/following-sibling::span/text()'
			xpath_close_price = '//b[contains(text(),"Close Price:")]/following-sibling::span/text()'
			xpath_close_date = '//b[contains(text(),"Close Date:")]/following-sibling::span/text()'
			xpath_key = './/div[@class="table-cell"]/b/text()'
			xpath_value = './/div[@class="table-cell"]/text()'

			raw_name = parser.xpath(xpath_head)
			key_stock_table =  parser.xpath(xpath_key_stock_table)
			raw_open_price = parser.xpath(xpath_open_price)
			raw_open_date = parser.xpath(xpath_open_date)
			raw_close_price = parser.xpath(xpath_close_price)
			raw_close_date = parser.xpath(xpath_close_date)
	
			company_name = raw_name[0].replace("Common Stock Quote & Summary Data","").strip() if raw_name else ''
			open_price =raw_open_price[0].strip() if raw_open_price else None
			open_date = raw_open_date[0].strip() if raw_open_date else None
			close_price = raw_close_price[0].strip() if raw_close_price else None
			close_date = raw_close_date[0].strip() if raw_close_date else None

			note = ""
			for i in key_stock_table:
				key = i.xpath(xpath_key)
				value = i.xpath(xpath_value)
				key = ''.join(key).strip() 
				value = ' '.join(''.join(value).split()) 
				note = note + key + ": "
				note = note + value + "; "
				
			nasdaq_data = [company_name, ticker, url, open_price, open_date, close_price, close_date, note]
			return nasdaq_data

		except Exception as e:
			print("Failed to process the request, Exception:%s"%(e))

def add_detail_toframe(nasdaq_data, detail):
	if detail.getVarCount() == 0:
		detail.addVarStr('company', 10)
		detail.addVarStr('ticker', 10)
		detail.addVarStr('url', 10)
		detail.addVarStr('open_price', 10)
		detail.addVarStr('open_date', 10)
		detail.addVarStr('close_price', 10)
		detail.addVarStr('close_date', 10)
		detail.addVarStrL('note')

	obs = detail.getObsTotal()
	detail.addObs(1)
	detail.store(None, obs, nasdaq_data)

detail = Frame.connect(framename)
data = get_nasdaq_detail(stockticker)
add_detail_toframe(data, detail)
