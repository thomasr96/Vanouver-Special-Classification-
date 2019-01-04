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
	name = 'vspec_img_crawl'
	allowed_domains = ['vancouverspecial.com']
	start_urls = ['http://vancouverspecial.com/?num=' + str(i) for i in range(1, picNum+1)]

	def parse(self, response):


		cur = response.request.url

		count = cur[len('http://vancouverspecial.com/?num='):]

		urllib.urlretrieve(str(website+response.css("img::attr(src)").extract()[0]), dir + str(count) + ".jpg")
		

		print("\n\n"+count+"\n\n")
