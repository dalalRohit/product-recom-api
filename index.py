from flask import Flask, render_template, request,jsonify
import requests
import subprocess

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template("home.html")


@app.route("/amazon")
def amazon():
    return render_template("amazon.html")


@app.route('/scrape-amazon',methods=['POST'])
def scrapeAmazon():
    data=request.json

    with open("{}".format(data['filename']), "w"):
        pass
    spider_name = "amazon"

    subprocess.check_output(['scrapy', 'crawl', spider_name,
        "-a","product={}".format(data['product']),
        "-a", "asin={}".format(data['asin']),
        "-o", "{}".format(data['filename'])])

    with open("{}".format(data['filename']),"r") as items_file:
        return items_file.read()

@app.route("/flipkart")
def flipkart():
    return render_template("flipkart.html")

# Scrape flipkart
@app.route("/scrape-flipkart", methods=["POST"])
def scrapeFlipkart():
    return "Hello POST /scrape-flipkart"

'''
@app.route('/test2',methods=['POST'])
def test2():

    data=request.json
    with open("{}".format(data['filename']), "w"):
        pass
    spider_name = "test"
    subprocess.check_output(['scrapy', 'crawl', spider_name,
        "-a","product={}".format(data['product']),
        "-a", "asin={}".format(data['asin']),
        "-o", "{}".format(data['filename'])])

    with open("{}".format(data['filename']),"r") as items_file:
        return items_file.read()
'''

if __name__ == '__main__':
    app.run(debug=True)
