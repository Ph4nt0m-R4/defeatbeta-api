import logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd

from defeatbeta_api.client.duckdb_client import get_duckdb_client
from defeatbeta_api.client.duckdb_conf import Configuration
from defeatbeta_api.client.hugging_face_client import HuggingFaceClient
from defeatbeta_api.data.news import News
from defeatbeta_api.data.transcripts import Transcripts
from defeatbeta_api.data.sql.sql_loader import load_sql
from defeatbeta_api.utils.const import stock_profile, stock_earning_calendar, stock_officers, \
    stock_split_events, stock_dividend_events, stock_tailing_eps, \
    stock_prices, stock_statement, stock_earning_call_transcripts, \
    stock_news, stock_revenue_breakdown, stock_shares_outstanding, stock_sec_filing


class Tickers:
    """
    Handle operations on multiple stock tickers efficiently with batch queries.
    
    This class performs aggregate queries on multiple symbols in a single operation,
    significantly reducing the number of database calls compared to creating 
    individual Ticker objects.
    
    Example:
        tickers = Tickers(['NVDA', 'AAPL', 'MSFT'])
        price_data = tickers.price()  # Single query for all prices
        info_data = tickers.info()    # Single query for all info
    """
    
    def __init__(self, tickers: List[str], http_proxy: Optional[str] = None, 
                 log_level: Optional[str] = logging.INFO, config: Optional[Configuration] = None):
        """
        Initialize Tickers with a list of symbols.
        
        Args:
            tickers: List of stock symbols (e.g., ['NVDA', 'AAPL', 'MSFT'])
            http_proxy: Optional HTTP proxy URL
            log_level: Logging level (default: logging.INFO)
            config: Optional DuckDB configuration
        """
        self.tickers = [t.upper() for t in tickers]
        self.http_proxy = http_proxy
        self.config = config
        self.duckdb_client = get_duckdb_client(http_proxy=self.http_proxy, log_level=log_level, config=config)
        self.huggingface_client = HuggingFaceClient()
        self.log_level = log_level
    
    # ========== Price & Market Data ==========
    
    def price(self) -> pd.DataFrame:
        """
        Get stock prices for all tickers with a single query.
        
        Returns:
            DataFrame with columns: symbol, report_date, open, high, low, close, volume
        """
        return self._query_multiple_data(stock_prices)
    
    def ttm_eps(self) -> pd.DataFrame:
        """Get trailing twelve months EPS for all tickers with a single query."""
        return self._query_multiple_data(stock_tailing_eps)
    
    def ttm_pe(self) -> pd.DataFrame:
        """
        Get trailing twelve months P/E ratio for all tickers with batch queries.
        
        Returns:
            DataFrame with price, EPS, and P/E ratio for all symbols
        """
        price_df = self.price()
        eps_df = self.ttm_eps()
        
        price_df['report_date'] = pd.to_datetime(price_df['report_date'])
        eps_df['report_date'] = pd.to_datetime(eps_df['report_date'])
        
        # Merge on both symbol and date for batch processing
        result_df = pd.merge_asof(
            price_df.sort_values(['symbol', 'report_date']),
            eps_df.sort_values(['symbol', 'report_date']),
            left_on=['symbol', 'report_date'],
            right_on=['symbol', 'report_date'],
            direction='backward'
        )
        
        result_df['ttm_pe'] = round(
            result_df['close'].astype(float) / result_df['tailing_eps'].astype(float),
            2
        )
        
        result_df = result_df[[
            'symbol',
            'report_date',
            'close',
            'tailing_eps',
            'ttm_pe'
        ]]
        
        result_df = result_df.rename(columns={
            'close': 'close_price',
            'tailing_eps': 'ttm_eps',
        })
        
        return result_df
    
    def shares(self) -> pd.DataFrame:
        """Get shares outstanding for all tickers with a single query."""
        return self._query_multiple_data(stock_shares_outstanding)
    
    # ========== Company Information ==========
    
    def info(self) -> pd.DataFrame:
        """Get company information for all tickers with a single query."""
        return self._query_multiple_data(stock_profile)
    
    def officers(self) -> pd.DataFrame:
        """Get officer information for all tickers with a single query."""
        return self._query_multiple_data(stock_officers)
    
    def sec_filing(self) -> pd.DataFrame:
        """Get SEC filing information for all tickers with a single query."""
        url = self.huggingface_client.get_url_path(stock_sec_filing)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_sec_filing_by_symbols", symbols=ticker_list, url=url)
        return self.duckdb_client.query(sql)
    
    # ========== Events & News ==========
    
    def calendar(self) -> pd.DataFrame:
        """Get earning calendar for all tickers with a single query."""
        return self._query_multiple_data(stock_earning_calendar)
    
    def splits(self) -> pd.DataFrame:
        """Get stock split events for all tickers with a single query."""
        return self._query_multiple_data(stock_split_events)
    
    def dividends(self) -> pd.DataFrame:
        """Get dividend events for all tickers with a single query."""
        return self._query_multiple_data(stock_dividend_events)
    
    def earning_call_transcripts(self) -> Transcripts:
        """Get earning call transcripts for all tickers with a single query."""
        df = self._query_multiple_data(stock_earning_call_transcripts)
        return Transcripts(df, self.log_level)
    
    def news(self) -> News:
        """Get news for all tickers with a single query."""
        url = self.huggingface_client.get_url_path(stock_news)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_news_by_symbols", symbols=ticker_list, url=url)
        return News(self.duckdb_client.query(sql))
    
    # ========== Financial Statements ==========
    
    def quarterly_income_statement(self) -> pd.DataFrame:
        """Get quarterly income statement for all tickers with a single query."""
        return self._statement('income_statement', 'quarterly')
    
    def annual_income_statement(self) -> pd.DataFrame:
        """Get annual income statement for all tickers with a single query."""
        return self._statement('income_statement', 'annual')
    
    def quarterly_balance_sheet(self) -> pd.DataFrame:
        """Get quarterly balance sheet for all tickers with a single query."""
        return self._statement('balance_sheet', 'quarterly')
    
    def annual_balance_sheet(self) -> pd.DataFrame:
        """Get annual balance sheet for all tickers with a single query."""
        return self._statement('balance_sheet', 'annual')
    
    def quarterly_cash_flow(self) -> pd.DataFrame:
        """Get quarterly cash flow for all tickers with a single query."""
        return self._statement('cash_flow', 'quarterly')
    
    def annual_cash_flow(self) -> pd.DataFrame:
        """Get annual cash flow for all tickers with a single query."""
        return self._statement('cash_flow', 'annual')
    
    # ========== Profitability Margins ==========
    
    def quarterly_gross_margin(self) -> pd.DataFrame:
        """Get quarterly gross margin for all tickers."""
        return self._generate_margin('gross', 'quarterly', 'gross_profit', 'gross_margin')
    
    def annual_gross_margin(self) -> pd.DataFrame:
        """Get annual gross margin for all tickers."""
        return self._generate_margin('gross', 'annual', 'gross_profit', 'gross_margin')
    
    def quarterly_operating_margin(self) -> pd.DataFrame:
        """Get quarterly operating margin for all tickers."""
        return self._generate_margin('operating', 'quarterly', 'operating_income', 'operating_margin')
    
    def annual_operating_margin(self) -> pd.DataFrame:
        """Get annual operating margin for all tickers."""
        return self._generate_margin('operating', 'annual', 'operating_income', 'operating_margin')
    
    def quarterly_net_margin(self) -> pd.DataFrame:
        """Get quarterly net margin for all tickers."""
        return self._generate_margin('net', 'quarterly', 'net_income_common_stockholders', 'net_margin')
    
    def annual_net_margin(self) -> pd.DataFrame:
        """Get annual net margin for all tickers."""
        return self._generate_margin('net', 'annual', 'net_income_common_stockholders', 'net_margin')
    
    def quarterly_ebitda_margin(self) -> pd.DataFrame:
        """Get quarterly EBITDA margin for all tickers."""
        return self._generate_margin('ebitda', 'quarterly', 'ebitda', 'ebitda_margin')
    
    def annual_ebitda_margin(self) -> pd.DataFrame:
        """Get annual EBITDA margin for all tickers."""
        return self._generate_margin('ebitda', 'annual', 'ebitda', 'ebitda_margin')
    
    def quarterly_fcf_margin(self) -> pd.DataFrame:
        """Get quarterly FCF margin for all tickers."""
        return self._generate_margin('fcf', 'quarterly', 'free_cash_flow', 'fcf_margin')
    
    def annual_fcf_margin(self) -> pd.DataFrame:
        """Get annual FCF margin for all tickers."""
        return self._generate_margin('fcf', 'annual', 'free_cash_flow', 'fcf_margin')
    
    # ========== Revenue Breakdown ==========
    
    def revenue_by_segment(self) -> pd.DataFrame:
        """Get revenue by business segment for all tickers with a single query."""
        return self._revenue_by_breakdown('segment')
    
    def revenue_by_geography(self) -> pd.DataFrame:
        """Get revenue by geographic region for all tickers with a single query."""
        return self._revenue_by_breakdown('geography')
    
    def revenue_by_product(self) -> pd.DataFrame:
        """Get revenue by product for all tickers with a single query."""
        return self._revenue_by_breakdown('product')
    
    # ========== Utility & Helper Methods ==========
    
    def get_by_symbol(self, symbol: str) -> 'Tickers':
        """
        Get a Tickers object for a single symbol.
        
        Args:
            symbol: Stock symbol to filter to
            
        Returns:
            Tickers object with single symbol
            
        Raises:
            ValueError: If symbol is not in the tickers list
        """
        if symbol.upper() in self.tickers:
            return Tickers([symbol], self.http_proxy, self.log_level, self.config)
        raise ValueError(f"Symbol {symbol} not in {self.tickers}")
    
    def to_dict(self) -> Dict[str, 'Tickers']:
        """
        Convert to dictionary of Tickers objects by symbol.
        
        Returns:
            Dictionary with symbols as keys and individual Tickers objects as values
        """
        return {
            symbol: Tickers([symbol], self.http_proxy, self.log_level, self.config)
            for symbol in self.tickers
        }
    
    # ========== Private Helper Methods ==========
    
    def _query_multiple_data(self, table_name: str) -> pd.DataFrame:
        """
        Query data for multiple symbols with a single query.
        
        Args:
            table_name: Name of the data table
            
        Returns:
            DataFrame with data for all symbols
        """
        url = self.huggingface_client.get_url_path(table_name)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_all_by_symbols", symbols=ticker_list, url=url)
        return self.duckdb_client.query(sql)
    
    def _statement(self, finance_type: str, period_type: str) -> pd.DataFrame:
        """
        Get financial statement for all tickers.
        
        Args:
            finance_type: Type of financial statement (income_statement, balance_sheet, cash_flow)
            period_type: Period type ('quarterly' or 'annual')
            
        Returns:
            DataFrame with financial statement data for all symbols
        """
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_statement_by_symbols", symbols=ticker_list, url=url, finance_type=finance_type, period_type=period_type)
        return self.duckdb_client.query(sql)
    
    def _revenue_by_breakdown(self, breakdown_type: str) -> pd.DataFrame:
        """
        Get revenue breakdown for all tickers.
        
        Args:
            breakdown_type: Type of breakdown ('segment', 'geography', or 'product')
            
        Returns:
            DataFrame with revenue breakdown for all symbols
        """
        url = self.huggingface_client.get_url_path(stock_revenue_breakdown)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_revenue_breakdown_by_symbols", symbols=ticker_list, url=url, breakdown_type=breakdown_type)
        return self.duckdb_client.query(sql)
    
    def _generate_margin(self, margin_type: str, period_type: str, 
                         metric_column: str, margin_column_name: str) -> pd.DataFrame:
        """
        Calculate margin for all tickers with batch query.
        
        Args:
            margin_type: Type of margin ('gross', 'operating', 'net', 'ebitda', 'fcf')
            period_type: Period type ('quarterly' or 'annual')
            metric_column: Name of the metric column
            margin_column_name: Name for the margin column in result
            
        Returns:
            DataFrame with margin calculations for all symbols
        """
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        
        # Map margin types to their item names in the database
        margin_item_map = {
            'gross': 'gross_profit',
            'operating': 'operating_income',
            'net': 'net_income_common_stockholders',
            'ebitda': 'ebitda',
            'fcf': 'free_cash_flow'
        }
        numerator_item = margin_item_map.get(margin_type, margin_type)
        
        sql = load_sql(
            "select_margin_for_symbols",
            symbols=ticker_list,
            url=url,
            numerator_item=numerator_item,
            margin_column=margin_column_name,
            finance_type_filter="",
            ttm_filter="",
            period_type=period_type
        )
        
        df = self.duckdb_client.query(sql)
        df['report_date'] = pd.to_datetime(df['report_date'])
        
        result_df = df[[
            'symbol',
            'report_date',
            numerator_item,
            'total_revenue',
            margin_column_name
        ]].copy()
        
        return result_df

    def roe(self) -> pd.DataFrame:
        """
        Get Return on Equity (ROE) for multiple symbols.
        
        Returns:
            pd.DataFrame: ROE data for each symbol with columns:
                - symbol
                - report_date
                - net_income_common_stockholders
                - beginning_stockholders_equity
                - ending_stockholders_equity
                - avg_equity
                - roe
        """
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_roe_by_symbols", symbols=ticker_list, url=url)
        result_df = self.duckdb_client.query(sql)
        result_df = result_df[[
            'symbol',
            'report_date',
            'net_income_common_stockholders',
            'beginning_stockholders_equity',
            'ending_stockholders_equity',
            'avg_equity',
            'roe'
        ]]
        return result_df

    def roa(self) -> pd.DataFrame:
        """
        Get Return on Assets (ROA) for multiple symbols.
        
        Returns:
            pd.DataFrame: ROA data for each symbol with columns:
                - symbol
                - report_date
                - net_income_common_stockholders
                - beginning_total_assets
                - ending_total_assets
                - avg_assets
                - roa
        """
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_roa_by_symbols", symbols=ticker_list, url=url)
        result_df = self.duckdb_client.query(sql)
        result_df = result_df[[
            'symbol',
            'report_date',
            'net_income_common_stockholders',
            'beginning_total_assets',
            'ending_total_assets',
            'avg_assets',
            'roa'
        ]]
        return result_df

    def roic(self) -> pd.DataFrame:
        """
        Get Return on Invested Capital (ROIC) for multiple symbols.
        
        Returns:
            pd.DataFrame: ROIC data for each symbol with columns:
                - symbol
                - report_date
                - ebit
                - tax_rate_for_calcs
                - nopat
                - beginning_invested_capital
                - ending_invested_capital
                - avg_invested_capital
                - roic
        """
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_roic_by_symbols", symbols=ticker_list, url=url)
        result_df = self.duckdb_client.query(sql)
        result_df = result_df[[
            'symbol',
            'report_date',
            'ebit',
            'tax_rate_for_calcs',
            'nopat',
            'beginning_invested_capital',
            'ending_invested_capital',
            'avg_invested_capital',
            'roic'
        ]]
        return result_df
    def ttm_revenue(self) -> pd.DataFrame:
        """
        Get Trailing Twelve Month (TTM) revenue for multiple symbols.
        
        Returns:
            pd.DataFrame: TTM revenue data for each symbol with columns:
                - symbol
                - report_date
                - ttm_total_revenue
                - report_date_2_revenue (JSON mapping of quarterly dates)
        """
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_ttm_revenue_by_symbols", symbols=ticker_list, ttm_revenue_url=url)
        return self.duckdb_client.query(sql)

    def ttm_fcf(self) -> pd.DataFrame:
        """
        Get Trailing Twelve Month (TTM) free cash flow for multiple symbols.
        
        Returns:
            pd.DataFrame: TTM FCF data for each symbol with columns:
                - symbol
                - report_date
                - ttm_free_cash_flow
                - report_date_2_fcf (JSON mapping of quarterly dates)
        """
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_ttm_fcf_by_symbols", symbols=ticker_list, ttm_fcf_url=url)
        return self.duckdb_client.query(sql)

    def ttm_net_income_common_stockholders(self) -> pd.DataFrame:
        """
        Get Trailing Twelve Month (TTM) net income for common stockholders for multiple symbols.
        
        Returns:
            pd.DataFrame: TTM net income data for each symbol with columns:
                - symbol
                - report_date
                - ttm_net_income
                - report_date_2_net_income (JSON mapping of quarterly dates)
        """
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_ttm_net_income_common_stockholders_by_symbols", symbols=ticker_list, ttm_net_income_url=url)
        return self.duckdb_client.query(sql)

    def news(self) -> News:
        """
        Get news articles for multiple symbols.
        
        Returns:
            News: News object containing articles for all symbols
        """
        url = self.huggingface_client.get_url_path(stock_news)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_news_by_symbols", symbols=ticker_list, url=url)
        return News(self.duckdb_client.query(sql))

    def quarterly_revenue_yoy_growth(self) -> pd.DataFrame:
        """Get quarterly revenue YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='total_revenue',
                       item_name='total_revenue',
                       period_type='quarterly',
                       finance_type='income_statement',
                       ttm_filter="AND report_date != 'TTM'")
        return self.duckdb_client.query(sql)

    def annual_revenue_yoy_growth(self) -> pd.DataFrame:
        """Get annual revenue YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='total_revenue',
                       item_name='total_revenue',
                       period_type='annual',
                       finance_type='income_statement',
                       ttm_filter='')
        return self.duckdb_client.query(sql)

    def quarterly_operating_income_yoy_growth(self) -> pd.DataFrame:
        """Get quarterly operating income YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='operating_income',
                       item_name='operating_income',
                       period_type='quarterly',
                       finance_type='income_statement',
                       ttm_filter="AND report_date != 'TTM'")
        return self.duckdb_client.query(sql)

    def annual_operating_income_yoy_growth(self) -> pd.DataFrame:
        """Get annual operating income YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='operating_income',
                       item_name='operating_income',
                       period_type='annual',
                       finance_type='income_statement',
                       ttm_filter='')
        return self.duckdb_client.query(sql)

    def quarterly_ebitda_yoy_growth(self) -> pd.DataFrame:
        """Get quarterly EBITDA YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='ebitda',
                       item_name='ebitda',
                       period_type='quarterly',
                       finance_type='income_statement',
                       ttm_filter="AND report_date != 'TTM'")
        return self.duckdb_client.query(sql)

    def annual_ebitda_yoy_growth(self) -> pd.DataFrame:
        """Get annual EBITDA YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='ebitda',
                       item_name='ebitda',
                       period_type='annual',
                       finance_type='income_statement',
                       ttm_filter='')
        return self.duckdb_client.query(sql)

    def quarterly_net_income_yoy_growth(self) -> pd.DataFrame:
        """Get quarterly net income YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='net_income_common_stockholders',
                       item_name='net_income_common_stockholders',
                       period_type='quarterly',
                       finance_type='income_statement',
                       ttm_filter="AND report_date != 'TTM'")
        return self.duckdb_client.query(sql)

    def annual_net_income_yoy_growth(self) -> pd.DataFrame:
        """Get annual net income YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='net_income_common_stockholders',
                       item_name='net_income_common_stockholders',
                       period_type='annual',
                       finance_type='income_statement',
                       ttm_filter='')
        return self.duckdb_client.query(sql)

    def quarterly_fcf_yoy_growth(self) -> pd.DataFrame:
        """Get quarterly free cash flow YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='free_cash_flow',
                       item_name='free_cash_flow',
                       period_type='quarterly',
                       finance_type='cash_flow',
                       ttm_filter="AND report_date != 'TTM'")
        return self.duckdb_client.query(sql)

    def annual_fcf_yoy_growth(self) -> pd.DataFrame:
        """Get annual free cash flow YoY growth for multiple symbols"""
        url = self.huggingface_client.get_url_path(stock_statement)
        ticker_list = ", ".join(f"'{t}'" for t in self.tickers)
        sql = load_sql("select_metric_calculate_yoy_growth_by_symbols",
                       symbols=ticker_list, url=url,
                       metric_name='free_cash_flow',
                       item_name='free_cash_flow',
                       period_type='annual',
                       finance_type='cash_flow',
                       ttm_filter='')
        return self.duckdb_client.query(sql)
