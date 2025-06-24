from vnstock3 import Vnstock
from vnstock3.explorer.tcbs.company import Company
from vnstock3.explorer.vci.financial import Finance
from vnstock3.explorer.misc.gold_price import *
from datetime import date
import os
if "ACCEPT_TC" not in os.environ:
    os.environ["ACCEPT_TC"] = "tôi đồng ý"
class StockEntity():
    def __init__(self,symbol,source):
        self.symbol = symbol
        self.source = source
    def dump_json(self,df):
        json_str = df.to_json(orient='records',force_ascii=False,date_format='iso')
        return json.loads(json_str)

class MarketStock(StockEntity):
    def __init__(self,symbol,source):
        super().__init__(symbol=symbol,source=source)
        self.stock = Vnstock().stock(symbol=self.symbol,source=self.source)
    def get_stock_by_group(self,group):
        if group != 'all':
            df = self.stock.listing.symbols_by_group(group)
        else:
            df = self.stock.listing.all_symbols()['ticker']
        return list(df)

    def get_price_board(self,group):
        lst_symbol = self.get_stock_by_group(group)
        return self.stock.trading.price_board(lst_symbol)
    def get_price_history(self,symbol,start,end):
        df = self.stock.quote.history(symbol=symbol,start=start,end=end)
        return df

    def get_companies_by_industry(self,industry_id):
        df = self.get_all_companies()
        df_industry = df[df['industry_id']==industry_id]
        return df_industry
    def get_top_gainers(self):
        df =self.get_all_companies()
        df['variation'] = df['close'] - df['open']
        df = df.sort_values('variation',ascending=False)
        return df.head(10)
    def get_top_losers(self):
        df =self.get_all_companies()
        df['variation'] = df['close'] - df['open']
        df = df.sort_values('variation')
        return df.head(10)
    def get_top_actives(self):
        df =self.get_all_companies()
        df = df.sort_values('volume',ascending=False)
        return df.head(10)
    def get_main_industries(self):
        df = self.get_all_companies()
        df = df[['industry_id_v2','industry']]
        df = df.drop_duplicates(subset=['industry_id_v2'])
        return df


class CompanyStock(StockEntity):

    def __init__(self,symbol,source):
        super(CompanyStock, self).__init__(symbol=symbol,source=source)
        self.company = Company(symbol=self.symbol)
    def over_view_company(self):
        return self.company.overview()
    def get_company_profile(self):
        return self.company.profile()
    def get_company_events(self):
        return self.company.events()
    def get_company_news(self):
        return self.company.news()
    def get_company_dividents(self):
        return self.company.dividends()
class StockFinance(StockEntity):
    def __init__(self,symbol,source,period,language):
        super(StockFinance, self).__init__(symbol=symbol,source=source)
        self.report = Finance(symbol=self.symbol,period=period)
        self.lang = language
    def get_company_balance_sheet(self):
        return self.report.balance_sheet(lang=self.lang)
    def get_income_statement(self):
        return self.report.income_statement(lang=self.lang)
    def get_cash_flow(self):
        return self.report.cash_flow(lang=self.lang)
    def get_ratio(self):
        return self.report.ratio(lang=self.lang)
class ForeignExchange(StockEntity):
    def __init__(self,symbol,source):
        super(ForeignExchange, self).__init__(symbol,source)
        self.fx = Vnstock.fx(symbol=self.symbol,source=self.source)
    def get_fx_history(self,start,end,interval):
        return self.fx.quote.history(start=start,end=end,interval=interval)
class Crypto(StockEntity):
    def __init__(self,symbol,source):
        super(Crypto, self).__init__(symbol=symbol,source=source)
        self.crypto = Vnstock().crypto(symbol=self.symbol,source=self.source)
    def get_crypto_history(self,start,end,interval):
        return self.crypto.quote.history(start=start,end=end,interval=interval)
class WorldIndex(StockEntity):
    def __init__(self,symbol,source):
        super(WorldIndex, self).__init__(symbol=symbol,source=source)
        self.index = Vnstock().world_index(symbol=self.symbol,source=self.source)
    def get_index_history(self,start,end,interval):
        return self.index.quote.history(start=start,end=end,interval=interval)

class Gold:
    @staticmethod
    def get_sjc_price():
        return sjc_gold_price()
    @staticmethod
    def get_bcmt_price():
        return btmc_goldprice()
