import scrapy
from ..items import ScrapperItem


class AmazonSpider(scrapy.Spider):

    def __init__(self,product='',asin='', **kwargs):
        self.url="https://www.amazon.in/{}/product-reviews/{}/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber=".format(product,asin)
        self.start_urls = []  # py36
        for i in range(1,100):
            self.start_urls.append(self.url+str(i))

        super().__init__(**kwargs)  # python3

    # Spider name
    name = 'amazon'

    # Domain names to scrape
    allowed_domains = ['amazon.in']


    # Defining a Scrapy parser
    def parse(self, response):
        items=ScrapperItem()

        # <div> of all reviews
        data = response.css('#cm_cr-review_list')

        # Usernames
        usernames = data.css('.a-profile-name::text').extract()

        # Profiles
        profiles = data.css('.a-profile::attr(href)').getall()
        for i in range(len(profiles)):
            profiles[i] = 'https://amazon.in/'+profiles[i]

        # User review date
        dates = data.css('.review-date::text').extract()

        # Collecting product star ratings
        star_ratings = data.css('.review-rating .a-icon-alt::text').extract()

        # Collecting user reviews
        comments = data.css('.review-text-content span::text').extract()

        # Collecting review title
        review_title = data.css('.review-title span::text').extract() #pending

        # verified
        verifieds = data.css('span.a-color-state.a-text-bold::text').extract()

        # helpful
        helpfuls = data.css('.cr-vote-text::text').extract()

        for i in range(0,len(usernames)):
            items['username']=usernames[i].strip()
            items['date']=dates[i].strip()
            items['profile']=profiles[i].strip()
            items['review_title']=review_title[i].strip()
            items['comments']=comments[i].strip()
            items['star_rating']=star_ratings[i].strip()
            items['verified']=verifieds[i].strip()
            items['helpful']=helpfuls[i].strip()


            yield items



