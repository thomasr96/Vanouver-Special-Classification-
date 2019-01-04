'''
scrapy startproject vanspec
cd vanspec
scrapy genspider vspecial vancouverspecial.com
scrapy crawl vspecial
'''
import scrapy
import urllib
import csv

website = 'http://vancouverspecial.com'
dir = "vancouver_special_images/vs_ "
picNum = 1240

class VspecialSpider(scrapy.Spider):
	name = 'address_crawler'
	allowed_domains = ['vancouverspecial.com']
	start_urls = ['http://vancouverspecial.com/?num=' + str(i) for i in range(1, picNum+1)]

	def parse(self, response):

		cur = response.request.url

		count = cur[len('http://vancouverspecial.com/?num='):]


		add_num = response.xpath('//div/text()').extract_first() 
		street = response.xpath('//a/text()').extract_first()
		print(count)

		with open('address_list.csv', 'a+') as myFile: 
			writer = csv.writer(myFile)
			writer.writerow([add_num + street + ' Vancouver, BC\n'])







