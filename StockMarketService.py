import pandas as pd

from StockEntity import MarketStock, CompanyStock, StockFinance


class StockMarketService:
    @staticmethod
    def get_companies_by_group(symbol, source, group):
        maket_stock = MarketStock(symbol=symbol, source=source)
        df = maket_stock.get_companies_info(group=group)
        return maket_stock.dump_json(df)

    @staticmethod
    def get_all_companies(symbol, source):
        maket_stock = MarketStock(symbol=symbol, source=source)
        df = maket_stock.get_all_companies()
        return maket_stock.dump_json(df)

    @staticmethod
    def get_companies_by_industry(industry_id, symbol, source):
        maket_stock = MarketStock(symbol=symbol, source=source)
        df = maket_stock.get_companies_by_industry(industry_id)
        return maket_stock.dump_json(df)

    @staticmethod
    def get_top_gainers(symbol, source):
        maket_stock = MarketStock(symbol=symbol, source=source)
        df = maket_stock.get_top_gainers()
        return maket_stock.dump_json(df)

    @staticmethod
    def get_top_losers(symbol, source):
        maket_stock = MarketStock(symbol=symbol, source=source)
        df = maket_stock.get_top_losers()
        return maket_stock.dump_json(df)

    @staticmethod
    def get_top_actives(symbol, source):
        maket_stock = MarketStock(symbol=symbol, source=source)
        df = maket_stock.get_top_actives()
        return maket_stock.dump_json(df)

    @staticmethod
    def get_price_boards(symbol, source, group):
        maket_stock = MarketStock(symbol=symbol, source=source)
        df = maket_stock.get_price_board(group)
        new_columns = []
        for column in df.columns:
            new_column = column[1]
            new_columns.append(new_column)
        df.columns = new_columns
        # join open data from xlsx file
        df_all = maket_stock.get_all_companies()
        df_all = df_all[['open', 'ticker']]
        df = df.merge(df_all, how='left', left_on="symbol", right_on="ticker")
        return maket_stock.dump_json(df)

    @staticmethod
    def get_industries(symbol, source):
        maket_stock = MarketStock(symbol=symbol, source=source)
        df = maket_stock.get_main_industries()
        return maket_stock.dump_json(df)

    @staticmethod
    def get_top_100_symbols(symbol, source):
        market_stock = MarketStock(symbol="VNMidCap", source=source)
        symbols = market_stock.get_stock_by_group(symbol)
        df = pd.DataFrame()
        df['ticker'] = symbols
        return market_stock.dump_json(df)


class CompanyService:
    @staticmethod
    def get_company_overview(symbol, source):
        company_obj = CompanyStock(symbol=symbol, source=source)
        df = company_obj.over_view_company()
        return company_obj.dump_json(df)

    @staticmethod
    def get_company_profile(symbol, source):
        company_obj = CompanyStock(symbol=symbol, source=source)
        df = company_obj.get_company_profile()
        return company_obj.dump_json(df)

    @staticmethod
    def get_company_events(symbol, source):
        company_obj = CompanyStock(symbol=symbol, source=source)
        df = company_obj.get_company_events()
        return company_obj.dump_json(df)

    @staticmethod
    def get_company_news(symbol, source):
        company_obj = CompanyStock(symbol=symbol, source=source)
        df = company_obj.get_company_news()
        return company_obj.dump_json(df)

    @staticmethod
    def get_company_dividends(symbol, source):
        company_obj = CompanyStock(symbol=symbol, source=source)
        df = company_obj.get_company_dividents()
        return company_obj.dump_json(df)

    @staticmethod
    def get_price_history(symbol, source, start, end):
        market_stock = MarketStock(symbol=symbol, source=source)
        df = market_stock.get_price_history(symbol=symbol, start=start, end=end)
        return market_stock.dump_json(df)

    @staticmethod
    def get_price_history_df(symbol, source, start='2000-01-01', end='2100-01-01'):
        market_stock = MarketStock(symbol=symbol, source=source)
        return market_stock.get_price_history(symbol=symbol, start=start, end=end)

    @staticmethod
    def get_all_history_data(symbol, source):
        market_stock = MarketStock(symbol=symbol, source=source)
        df = market_stock.get_price_history(symbol=symbol, start='2000-01-01', end='2100-01-01')
        return market_stock.dump_json(df)


class FinancialService:
    @staticmethod
    def get_financial_income_statement(symbol, source, period, language):
        financial_obj = StockFinance(symbol, source, period, language)
        df = financial_obj.get_income_statement()
        df = df.iloc[:,
             [0, 1, 3, 5, 6, 8, 10, 11, 14, 16, 18, 20, 22, 24, 26, 29, 31, 33, 34, 35, 36, 40, 50, 52, 54, 56, 58]]
        df = df.rename(columns={"ticker": "symbol", "yearReport": "report_year"})
        df = df.rename(columns={"Minority Interest": "minor_interest", "Net Profit/Loss before tax": "net_profit"})
        df = df.rename(
            columns={"Dividends received": "dividend_receive", "Provision for credit losses": "provision_credit_lost"})
        df = df.rename(columns={"Revenue YoY (%)": "revenue_percent", "Revenue (Bn. VND)": "revenue"})
        df = df.rename(columns={"Financial Income": "finance_income", "Interest Expenses": "interest_expense"})

        df = df.rename(columns={"Net Sales": "net_sales", "Gross Profit": "gross_profit"})
        df = df.rename(
            columns={"Financial Expenses": "financial_expenses", "General & Admin Expenses": "general_admin_expenses"})
        df = df.rename(
            columns={"Operating Profit/Loss": "operating_profit", "Other Income/Expenses": "other_income_expenses"})
        df = df.rename(columns={"Selling Expenses": "selling_expenses", "Sales": "sales"})
        df = df.rename(columns={"Sales deductions": "sales_deductions",
                                "Gain/(loss) from joint ventures": "gain_from_joint_ventures"})

        df = df.rename(columns={"Other income": "other_income", "Net Interest Income": "net_interest_income"})
        df = df.rename(
            columns={"Total operating revenue": "total_operating_revenue", "Profit before tax": "profit_before_tax"})
        df = df.rename(columns={"Business income tax - current": "income_tax_current",
                                "Net Profit For the Year": "net_profit_for_the_year"})
        df = df.rename(columns={"EPS_basis": "eps_basis"})

        return financial_obj.dump_json(df)

    @staticmethod
    def get_financial_cash_flow(symbol, source, period, language):
        financial_obj = StockFinance(symbol, source, period, language)
        df = financial_obj.get_cash_flow()
        df = df.iloc[:,
             [0, 1, 3, 5, 7, 8, 9, 10, 12, 13, 16, 20, 24, 25, 28, 29, 33, 35, 36, 37, 38, 40, 41, 43]]
        df = df.rename(columns={"ticker": "symbol", "yearReport": "report_year"})
        df = df.rename(columns={"Cash and cash equivalents": "cash",
                                "Profit/Loss from disposal of fixed assets": "profit_disposal_fixed_assets"})
        df = df.rename(columns={"Profit/Loss from investing activities": "profit_investing_activities",
                                "Interest Expense": "interest_expense"})
        df = df.rename(columns={"Interest income and dividends": "interest_income_and_dividends",
                                "Increase/Decrease in receivables": "increase_decrease_in_receivables"})
        df = df.rename(columns={"Increase/Decrease in inventories": "increase_decrease_in_inventories",
                                "Increase/Decrease in payables": "increase_decrease in payables"})

        df = df.rename(columns={"Interest paid": "interest_paid",
                                "Net cash inflows/outflows from operating activities": "cash_from_operating_activities"})
        df = df.rename(columns={"Payments for share repurchases": "payments_for_share_epurchases",
                                "Proceeds from borrowings": "proceeds_from_borrowings"})
        df = df.rename(columns={"Dividends paid": "dividends_paid",
                                "Net increase/decrease in cash and cash equivalents": "net_change_in_cash"})
        df = df.rename(columns={"Profits from other activities": "profits_from_other_activities",
                                "Net Cash Flows from Operating Activities before BIT": "net_cash_flows_from_operating_activities_before_BIT"})
        df = df.rename(columns={"Payment from reserves": "payment_from_reserves",
                                "Purchase of fixed assets": "purchase_of_fixed_assets"})

        df = df.rename(columns={"Gain on Dividend": "gain_on_dividend",
                                "Increase in charter captial": "increase_in_charter_captial"})
        df = df.rename(columns={"Cash flows from financial activities": "cash_flows_from_financial_activities",
                                "Loans granted, purchases of debt instruments (Bn. VND)": "loans_granted"})
        return financial_obj.dump_json(df)

    @staticmethod
    def get_company_balance_sheet(symbol, source, period, language):
        financial_obj = StockFinance(symbol, source, period, language)
        df = financial_obj.get_company_balance_sheet()
        df = df.iloc[:,
             [0, 1, 3, 5, 9, 11, 12, 13, 16, 19, 20, 23, 27, 30, 31, 32, 33, 40, 44, 45, 46, 49, 50, 52, 53, 55, 58,
              67, 71, 72]]
        df = df.rename(columns={"ticker": "symbol", "yearReport": "report_year"})
        df = df.rename(
            columns={"Investment in properties": "invest_in_prop", "Budget sources and other funds": "budget_sources"})
        df = df.rename(columns={"MINORITY INTERESTS": "minority_interest", "Goodwill": "good_will"})
        df = df.rename(
            columns={"Balances with the SBV": "balances_with_sbv", "Trading Securities": "trading_securities"})
        df = df.rename(columns={"Derivatives and other financial liabilities": "derivatives",
                                "Loans and advances to customers, net": "loan"})

        df = df.rename(columns={"Investment Securities": "investment_securities",
                                "Investment in joint ventures": "investment_joint_ventures"})
        df = df.rename(columns={"Tangible fixed assets": "tangible_fixed_assets", "Leased assets": "leased_assets"})
        df = df.rename(columns={"Intagible fixed assets": "intagible_fixed_assets", "Other Assets": "other_assets"})
        df = df.rename(columns={"Capital": "capital", "CURRENT ASSETS (Bn. VND)": "current_assets"})
        df = df.rename(columns={"Cash and cash equivalents (Bn. VND)": "cash",
                                "Short-term investments (Bn. VND)": "short_term_investments"})

        df = df.rename(columns={"Short-term loans receivables (Bn. VND)": "short_term_loans_receivables",
                                "Inventories, Net (Bn. VND)": "inventories"})
        df = df.rename(
            columns={"LONG-TERM ASSETS (Bn. VND)": "long_term_assets", "TOTAL RESOURCES (Bn. VND)": "total_resources"})
        df = df.rename(columns={"Investment and development funds (Bn. VND)": "investment_development_funds",
                                "Capital and reserves (Bn. VND)": "capital_and_reserves"})
        df = df.rename(columns={"TOTAL ASSETS (Bn. VND)": "total_assets",
                                "Long-term investments (Bn. VND)": "long_term_investments"})
        df = df.rename(columns={"Fixed assets (Bn. VND)": "fixed_assets",
                                "Long-term trade receivables (Bn. VND)": "long_term_trade_receivables"})

        return financial_obj.dump_json(df)

    @staticmethod
    def get_company_financial_ratio(symbol, source, period, language):
        financial_obj = StockFinance(symbol, source, period, language)
        df = financial_obj.get_ratio()
        df.columns = ['{}_{}'.format(x[0], x[1]) for x in df.columns]
        df.columns = [x.replace("Chỉ tiêu định giá", "") for x in df.columns]
        df.columns = [x.replace("Chỉ tiêu khả năng sinh lợi", "") for x in df.columns]
        df.columns = [x.replace("Chỉ tiêu thanh khoản", "") for x in df.columns]
        df.columns = [x.replace("Chỉ tiêu hiệu quả hoạt động", "") for x in df.columns]
        df.columns = [x.replace("Chỉ tiêu cơ cấu nguồn vốn", "") for x in df.columns]
        df.columns = [x.lower() for x in df.columns]
        df.columns = [x.replace('_', "") for x in df.columns]
        df.columns = [x.replace('(%)', "") for x in df.columns]
        df.columns = [x.replace('(bn. vnd)', "") for x in df.columns]
        df.columns = [x.replace('(mil. shares)', "") for x in df.columns]
        df.columns = [x.replace('(st+lt borrowings)/', "") for x in df.columns]
        df.columns = [x.replace('(vnd)', "") for x in df.columns]
        df.columns = [x.replace("'", "") for x in df.columns]
        df.columns = [x.replace("/", " ") for x in df.columns]
        df.columns = [x.replace("-", " ") for x in df.columns]

        df.columns = [x.strip() for x in df.columns]
        df.columns = [x.replace(" ", "_") for x in df.columns]

        return financial_obj.dump_json(df)











