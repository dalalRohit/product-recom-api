import scrapy
from ..FlipkartItems import ScrapperItem


class FKSpider(scrapy.Spider):

    def __init__(self,product='',asin='', **kwargs):
        self.url="https://www.flipkart.com/mi-a3-kind-grey-64-gb/product-reviews/itm280ef520ac1d9?pid=MOBFJM4ZZW6NTSZH&lid=LSTMOBFJM4ZZW6NTSZHU6DTVH&marketplace=FLIPKART&page="
        self.start_urls = []  # py36
        for i in range(1,3):
            self.start_urls.append(self.url+str(i))

        super().__init__(**kwargs)  # python3

    name = "flipkart"

    allowed_domains = ['flipkart.com']

    # Parse function
    def parse(self, response):
        items=ScrapperItem()

        #divs
        div=response.css('._1PBCrt')

        # username and dates
        username_dates=div.css('p._3LYOAd::text').extract()
        print("Length of username_dates: ",len(username_dates))

        #certified buyer and location
        buyer_location=div.css('._19inI8 span::text').extract()
        print("Length of buyer_location: ",len(buyer_location))


        #content
        content=div.css('.qwjRop div div::text').extract()
        print("Length of content: ",len(content))


        # for i in range(0,len(username_dates)):
        #     print(username_dates[i])









