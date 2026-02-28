import logging
import sys
import time
import pandas as pd
import requests
from lxml import etree

logger = logging.getLogger(__name__)

session = requests.Session()
HEADERS = {
    "User-Agent": "DefeatBeta contact@defeatbeta.com",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"
}
session.headers.update(HEADERS)

TRANSACTION_CODE_MAP = {
    "P": "Open Market Purchase",
    "S": "Open Market Sale",
    "V": "Voluntary Report",
    "A": "Grant / Award",
    "D": "Disposition to Issuer",
    "F": "Tax Withholding Sale",
    "G": "Gift",
    "C": "Conversion / Exercise",
    "M": "Option Exercise"
}

def classify_transaction(code):
    return TRANSACTION_CODE_MAP.get(code, "Other")

def parse_transactions(section, derivative=False):
    transactions = []
    for txn in section:
        code = txn.findtext(".//transactionCoding/transactionCode")
        transaction = {
            "type": "Derivative" if derivative else "Non-Derivative",
            "security_title": txn.findtext(".//securityTitle/value"),
            "transaction_date": txn.findtext(".//transactionDate/value"),
            "transaction_code": code,
            "transaction_nature": classify_transaction(code),
            "shares": txn.findtext(".//transactionAmounts/transactionShares/value"),
            "price_per_share": txn.findtext(".//transactionAmounts/transactionPricePerShare/value"),
            "acquired_disposed": txn.findtext(".//transactionAmounts/transactionAcquiredDisposedCode/value"),
            "post_transaction_shares": txn.findtext(".//postTransactionAmounts/sharesOwnedFollowingTransaction/value"),
            "ownership_type": txn.findtext(".//ownershipNature/directOrIndirectOwnership/value"),
        }
        transactions.append(transaction)
    return transactions

def parse_form4_xml(cik: str, accession: str):
    try:
        # SEC URLs do not use dashes in the accession number folder name
        accession_clean = accession.replace("-", "")
        cik_no_zero = str(int(cik))
        base_path = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zero}/{accession_clean}"

        time.sleep(0.15) 
        index_url = f"{base_path}/index.json"
        r = session.get(index_url)

        if r.status_code != 200:
            return None

        index_data = r.json()
        files = index_data.get("directory", {}).get("item", [])

        xml_file = None
        for f in files:
            name = f.get("name", "")
            if name.endswith(".xml") and "primary_doc.xml" not in name and "xsl" not in name.lower():
                xml_file = name
                break

        if not xml_file:
            return None

        time.sleep(0.15) 
        xml_url = f"{base_path}/{xml_file}"
        r_xml = session.get(xml_url)

        if r_xml.status_code != 200:
            return None

        root = etree.fromstring(r_xml.content)
        transactions = []

        non_deriv = root.findall(".//nonDerivativeTransaction")
        deriv = root.findall(".//derivativeTransaction")

        transactions.extend(parse_transactions(non_deriv, derivative=False))
        transactions.extend(parse_transactions(deriv, derivative=True))

        reporting_owner = root.find(".//reportingOwner")
        owner_name = reporting_owner.findtext(".//rptOwnerName") if reporting_owner is not None else None

        return {
            "reporting_owner": owner_name,
            "transactions": transactions
        }

    except Exception:
        logger.exception("Failed to parse Form 4 XML for accession %s", accession)
        return None

def extract_insider_trades_from_df(filings_df: pd.DataFrame, limit: int = None, start_date: str = None) -> pd.DataFrame:
    """
    Takes the DataFrame output of Ticker.sec_filing() and extracts the actual Form 4 data.
    
    Args:
        filings_df: DataFrame from Ticker.sec_filing()
        limit: Maximum number of transaction rows to return (not filings).
               If None or 0, returns all transactions. Pass a large number to get all filings' transactions.
        start_date: ISO date string (``'YYYY-MM-DD'``). Only filings on or after this date are included.
                    If None, no date filtering is applied.
    """
    all_records = []

    # Filter strictly to Form 4 filings (Insider Trading)
    form4_df = filings_df[filings_df['form_type'] == '4'].copy()

    # Sort the filings from NEWEST to OLDEST before parsing
    form4_df['filing_date'] = pd.to_datetime(form4_df['filing_date'], errors='coerce')
    form4_df = form4_df.sort_values(by='filing_date', ascending=False)
    
    # Filter by start_date if provided
    if start_date:
        start_date_dt = pd.to_datetime(start_date, errors='coerce')
        if pd.notna(start_date_dt):
            form4_df = form4_df[form4_df['filing_date'] >= start_date_dt]
    
    if form4_df.empty:
        logger.info("No Form 4 filings found in the provided DataFrame.")
        return pd.DataFrame()

    # Loop through the rows DuckDB gave us
    # We parse ALL relevant filings, then limit rows AFTER processing
    for index, row in form4_df.iterrows():
        symbol = row['symbol']
        cik = str(row['cik'])
        accession = row['accession_number']
        
        parsed = parse_form4_xml(cik, accession)
        
        if parsed and parsed.get("transactions"):
            for txn in parsed["transactions"]:
                record = {
                    "symbol": symbol,
                    "reporting_owner": parsed["reporting_owner"],
                    "accession_number": accession,
                    "filing_date": row['filing_date'],  # Inject the filing date
                    **txn
                }
                all_records.append(record)
                
                # Update progress line in place if limit is specified
                if limit and limit > 0:
                    progress_str = f"Parsing Form 4 filings... ({len(all_records)}/{limit})"
                    sys.stdout.write(f"\r{progress_str}")
                    sys.stdout.flush()
                    if len(all_records) >= limit:
                        break
            
            # Break outer loop if we've hit the limit
            if limit and limit > 0 and len(all_records) >= limit:
                break
    
    # Clear the progress line
    if limit and limit > 0:
        sys.stdout.write("\r" + " " * 60 + "\r")  # Clear the line
        sys.stdout.flush()

    df = pd.DataFrame(all_records)
    
    if not df.empty:
        # Convert numeric columns to floats
        df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
        df['price_per_share'] = pd.to_numeric(df['price_per_share'], errors='coerce')
        df['post_transaction_shares'] = pd.to_numeric(df['post_transaction_shares'], errors='coerce')
        
        # Convert date strings to actual datetime objects
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df['filing_date'] = pd.to_datetime(df['filing_date'], errors='coerce')
        
        # Sort chronologically by transaction_date (newest first), then filing_date (newest first)
        # Handle NaT (Not a Time) values by putting them at the end
        df = df.sort_values(
            by=["symbol", "transaction_date", "filing_date"], 
            ascending=[True, False, False],
            na_position='last'
        ).reset_index(drop=True)
        
        # Apply the limit to transaction rows (not filings)
        if limit and limit > 0:
            df = df.head(limit)
        
        logger.info("Parsed %d insider trading transactions", len(df))
        
    return df
