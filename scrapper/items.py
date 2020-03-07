# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class ScrapperItem(scrapy.Item):
    # define the fields for your item here like:
    username = scrapy.Field()
    profile = scrapy.Field()
    date = scrapy.Field()
    star_rating = scrapy.Field()
    comments = scrapy.Field()
    review_title = scrapy.Field()
    verified = scrapy.Field()
    helpful = scrapy.Field()

