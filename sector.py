# Stock class to encapsulate stock data with ticker information.
class Stock:
    def __init__(self, ticker, cusip, desc):
        self.ticker = ticker
        self.cusip = cusip
        self.desc = desc
        
    def get_data(self):
        if not self.data:
            raise Exception("Stock dataframe has not been set")
        return self.data
    
    def __repr__(self):
        return str(self.data)

def get_ticker_dict(t):
    dic = {}
    for desc, tkr, _ in t:
        dic[tkr] = desc
    return dic

def get_desc_dict(t):
    dic = {}
    for desc, tkr, _ in t:
        dic[desc] = tkr
    return dic
    
sectors = [('Communication Services', 'XLC', '81369Y85'), ('Consumer Discretionary', 'XLY', '81369Y40'), ('Consumer Staples', 'XLP', '81369Y30'),
           ('Energy', 'XLE', '81369Y50'), ('Financials', 'XLF', '81369Y60'), ('Health Care', 'XLV', '81369Y20'), ('Industrials', 'XLI', '81369Y70'),
           ('Materials', 'XLB', '81369Y10'), ('Real Estate', 'XLRE', '81369Y86'), ('Technology', 'XLK', '81369Y80'), ('Utilities', 'XLU', '81369Y88')]
