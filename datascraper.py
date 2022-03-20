from sec_api import ExtractorApi, QueryApi

api_key = '3b3dec900e3dc17fff9cf867cfb145c9d3b8040544b33f6108ae960e5233c829'

class DataScraper:
    def __init__(self):
        self.extractorApi = ExtractorApi(api_key)
        self.queryApi = QueryApi(api_key=api_key)
        self.urls = {}

    # send query to get most recent form 10-Ks with a non-empty ticker, save them in a dictionary (self.urls)
    def get_urls(self, seen_companies = set(), counter = 0):
        # I can only query 50 at a time, so to query new form 10-Ks, "from" is set to increment with counter
        query = {
            "query": {
                "query_string": {
                    "query": "formType:\"10-K\" AND ticker:[A TO Z]"
                }
            },
            "from": counter * 50,
            "size": "50",
            "sort": [
                {
                    "filedAt": {
                        "order": "desc"
                    }
                }
            ]
        }
        filings = self.queryApi.get_filings(query)

        for filing in filings['filings']:
            if filing['ticker'] not in seen_companies: # check if company is already seen (ie. if a company submits a form 10-K, and then shortly afterwards submits an amended form)
                self.urls[filing['ticker']] = {'cik': filing['cik'], 'name': filing['companyName'], 'url': filing['linkToFilingDetails']}
                seen_companies.add(filing['ticker'])

    # iterates through self.urls and sends a query to extractorApi with each url to get the text from section 1, then writes it to a file with the ticker name as file name
    def get_text(self):
        def write_to_file():
            nonlocal filing, section_text
            f = open('data/' + filing + '.txt', 'w')
            f.write(section_text)
            f.close()

        for filing in self.urls:
            section_text = self.extractorApi.get_section(self.urls[filing]['url'], "1", "text")
            write_to_file()


    def process(self):
        for x in range(2): # can modify range(x) to get more (ie. x * 50) form 10-Ks, but a free API key only allows 100 queries
            self.get_urls(counter=x)
        self.get_text()


if __name__ == '__main__':
    ds = DataScraper()
    ds.process()
    print('finished')