from dataclasses import dataclass


AT_TABLE = "tradingdata.tt_audittrail_view"
AT_TICKER = "audit"
AT_DTYPE = "tt_audittrail"
AT_BEAUTIFULDATA_NAME = AT_TICKER + "_" + AT_DTYPE


@dataclass
class ATColumns:
    """Class to specify column names from AuditTrail"""

    TIME_SENT: str = "time_sent"
    EXCHANGE_TIME: str = "exchange_time"
    EXCHANGE: str = "exchange"
    EMAIL: str = "originator_email"
    ORDER_PRICE: str = "order_price"
    ORDER_PRICE_TICKS: str = "order_price_ticks"
    INSTRUMENT: str = "instrument"
    FILL_SIZE: str = "fill_size"
    FILL_TYPE: str = "fill_type"
    PRODUCT: str = "product"
    EXEC_TYPE: str = "execution_type"
    EXCH_TRANS_ID: str = "exchange_transaction_id"
    TT_PARENT_ID: str = "tt_parent_id"
    TT_ORDER_ID: str = "tt_order_id"
    PRODUCT_TYPE: str = "product_type"
    MATURITY_DATE: str = "maturity_date"
    DTIME: str = "dtime"
    MESSAGE_TYPE: str = "message_type"
    EXECUTION_TYPE: str = "execution_type"
    ACCOUNT: str = "account"
    ORIGINAL_TIME: str = "original_time"
    PASSIVE_AGRESSIVE: str = "passive_agressive"
    MATURITY_DATE: str = "maturity_date"
    CLEARING_DATE: str = "clearing_date"
    BUY_SELL: str = "buy_sell"
    TEXT_TT: str = "text_tt"
    DF_ROW_ID: str = "df_row_id"
    ORDER_ROUTE: str = "order_route"
    EXECUTED_SIZE: str = "executed_size"
    ORDER_SIZE: str = "order_size"


@dataclass
class QuotesColumns:
    DTIME: str = "dtime"
    QUOTE_ID: str = "quote_id"
    BID_PRICE: str = "bid_price"
    BID_SIZE: str = "bid_size"
    ASK_PRICE: str = "ask_price"
    ASK_SIZE: str = "ask_size"


@dataclass
class TradesColumns:
    DTIME: str = "dtime"
    EXCHANGE_TRADE_ID: str = "exch_trade_id"
    TRADE_PRICE: str = "trade_price"
    TRADE_SIZE: str = "trade_size"
    AGGRESSOR: str = "aggressor"


@dataclass
class TTColumns:
    """
    Columns names for audittrails directly exported from TT
    """

    PRICE: str = "Price"
    EXEC_TYPE: str = "ExecType"
    TIME: str = "Time"
    EXCH_DATE: str = "ExchDate"
    EXCH_TIME: str = "ExchTime"
    TYPE: str = "Type"
    CONTRACT: str = "Contract"


AT_TABLE_PKEYS = [ATColumns.DTIME, ATColumns.ACCOUNT, ATColumns.DF_ROW_ID]
TRADES_PKEYS = [TradesColumns.DTIME, TradesColumns.EXCHANGE_TRADE_ID]
QUOTES_PKEYS = [QuotesColumns.DTIME, QuotesColumns.QUOTE_ID]


@dataclass
class ETypes:
    """Class to specify message types from AuditTrail"""

    NEW: str = "NEW"
    TRADE: str = "TRADE"
    RESTATED: str = "RESTATED"
    REPLACED: str = "REPLACED"
    CANCELED: str = "CANCELED"


QUOTE_ORDERS = [ETypes.NEW, ETypes.REPLACED, ETypes.RESTATED, ETypes.CANCELED]


@dataclass
class ExchangeSide:
    BID: str = "bid"
    ASK: str = "ask"


@dataclass
class OTypes:
    BUY: str = "BUY"
    SELL: str = "SELL"


ORDER_TYPE_TO_SIDE = {OTypes.BUY: ExchangeSide.BID, OTypes.SELL: ExchangeSide.ASK}


@dataclass
class SynchMode:
    TRADES: str = "trades"
    QUOTES: str = "quotes"
    BOTH: str = "both"


@dataclass
class MatchingColumns:
    TRADES_OFFSETS: str = f"exch_synch_offset_{SynchMode.TRADES}"
    QUOTES_OFFSETS: str = f"exch_synch_offset_{SynchMode.QUOTES}"
    EXCH_TIME: str = "exch_time_audit_trail"
    AT_DTIME: str = "dtime_audit_trail"
    ROW_ID_AT: str = "row_id_audit_trail"
    T2T_DTIME: str = "dtime_quote"
    ID_QUOTE: str = "id_quote"
    EXACT: str = "exact"


# assuming first leg is active
SPREAD_DEFINITIONS = r"\\arfimasv04\ArfimaSpreads\AutoExports"
EXCEL_SPREAD_DATE_FROMAT = "%Y%m%d"
