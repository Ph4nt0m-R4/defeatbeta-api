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

        # Fetch index with retry on rate limiting
        max_retries = 3
        for attempt in range(max_retries):
            time.sleep(0.5 + (attempt * 0.5))  # Start at 0.5s, backoff 0.5s per retry
            index_url = f"{base_path}/index.json"
            r = session.get(index_url)

            if r.status_code == 429:
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            elif r.status_code != 200:
                return None
            else:
                break  # Success

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

        # Fetch XML with retry on rate limiting
        for attempt in range(max_retries):
            time.sleep(0.5 + (attempt * 0.5))  # Start at 0.5s, backoff 0.5s per retry
            xml_url = f"{base_path}/{xml_file}"
            r_xml = session.get(xml_url)

            if r_xml.status_code == 429:
                if attempt < max_retries - 1:
                    continue
                else:
                    return None
            elif r_xml.status_code != 200:
                return None
            else:
                break  # Success

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

def _finalize_df(df: pd.DataFrame, limit: int = None) -> pd.DataFrame:
    """Apply type conversions, sorting, and optional row limit to a transactions DataFrame."""
    if df.empty:
        return df

    # Convert numeric columns to floats
    for col in ('shares', 'price_per_share', 'post_transaction_shares'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert date strings to actual datetime objects
    for col in ('transaction_date', 'filing_date'):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Sort chronologically: newest first
    sort_cols = [c for c in ("symbol", "transaction_date", "filing_date") if c in df.columns]
    ascending = [True if c == "symbol" else False for c in sort_cols]
    df = df.sort_values(by=sort_cols, ascending=ascending, na_position='last').reset_index(drop=True)

    # Apply the limit to transaction rows (not filings)
    if limit and limit > 0:
        df = df.head(limit)

    return df


def _parse_accessions(form4_df: pd.DataFrame, accessions_to_parse: set, limit: int = None) -> pd.DataFrame:
    """Parse SEC Form 4 XML for a specific set of accession numbers.

    Args:
        form4_df: DataFrame of Form 4 filings (must contain symbol, cik, accession_number, filing_date).
        accessions_to_parse: Set of accession_number strings to fetch from SEC.
        limit: Optional cap on transaction rows (for progress display only).

    Returns:
        DataFrame with the newly parsed transactions (unsorted, raw types).
    """
    target_df = form4_df[form4_df['accession_number'].isin(accessions_to_parse)]
    if target_df.empty:
        return pd.DataFrame()

    all_records = []
    for _, row in target_df.iterrows():
        parsed = parse_form4_xml(str(row['cik']), row['accession_number'])

        if parsed is None or not parsed.get("transactions"):
            continue

        for txn in parsed["transactions"]:
            record = {
                "symbol": row['symbol'],
                "reporting_owner": parsed["reporting_owner"],
                "accession_number": row['accession_number'],
                "filing_date": row['filing_date'],
                **txn,
            }
            all_records.append(record)

            # Progress indicator
            if limit and limit > 0:
                sys.stdout.write(f"\rParsing Form 4 filings... ({len(all_records)}/{limit})")
                sys.stdout.flush()
                if len(all_records) >= limit:
                    break

        if limit and limit > 0 and len(all_records) >= limit:
            break

    if limit and limit > 0:
        sys.stdout.write("\r" + " " * 60 + "\r")
        sys.stdout.flush()

    return pd.DataFrame(all_records)


def extract_insider_trades_from_df(
    filings_df: pd.DataFrame,
    limit: int = None,
    start_date: str = None,
    cache=None,
) -> pd.DataFrame:
    """
    Takes the DataFrame output of Ticker.sec_filing() and extracts the actual Form 4 data.

    When *cache* (an :class:`InsiderTradesCache` instance) is supplied the
    function will:

    1. Check which accession numbers are already cached.
    2. Parse only the **new** ones from SEC.
    3. Merge the results, persist to cache, and return.

    Args:
        filings_df: DataFrame from Ticker.sec_filing()
        limit: Maximum number of transaction rows to return (not filings).
               If None or 0, returns all transactions.
        start_date: ISO date string (``'YYYY-MM-DD'``). Only filings on or
                    after this date are included.
        cache: Optional :class:`InsiderTradesCache` instance for disk caching.
    """
    # ── 1. Filter to Form 4 filings ──────────────────────────────────
    form4_df = filings_df[filings_df['form_type'] == '4'].copy()
    if form4_df.empty:
        return pd.DataFrame()

    form4_df['filing_date'] = pd.to_datetime(form4_df['filing_date'], errors='coerce')
    form4_df = form4_df.sort_values(by='filing_date', ascending=False)

    if start_date:
        start_date_dt = pd.to_datetime(start_date, errors='coerce')
        if pd.notna(start_date_dt):
            form4_df = form4_df[form4_df['filing_date'] >= start_date_dt]

    if form4_df.empty:
        logger.info("No Form 4 filings found in the provided DataFrame.")
        return pd.DataFrame()

    # Set of accession numbers the caller needs
    needed_accessions = set(form4_df['accession_number'].unique())

    # ── 2. Cache look-up ─────────────────────────────────────────────
    cached_df = None
    cached_accessions: set = set()

    if cache is not None:
        cached_accessions = cache.get_cached_accessions()
        new_accessions = needed_accessions - cached_accessions

        if not new_accessions:
            # Full cache hit — everything we need is already on disk
            cached_df = cache.load()
            if cached_df is not None:
                logger.info("Insider-trades cache HIT for %s (%d accessions)",
                            cache.ticker, len(needed_accessions))
                # Filter cached data to the requested accession numbers & date range
                result = cached_df[cached_df['accession_number'].isin(needed_accessions)]
                return _finalize_df(result, limit)
            # Cache file corrupted / missing — fall through to full parse
            new_accessions = needed_accessions
    else:
        new_accessions = needed_accessions

    # ── 3. Parse only the new accession numbers from SEC ─────────────
    logger.info("Parsing %d new Form 4 filing(s) from SEC for %s",
                len(new_accessions),
                form4_df['symbol'].iloc[0] if 'symbol' in form4_df.columns else "?")

    new_df = _parse_accessions(form4_df, new_accessions, limit=limit)

    # ── 4. Merge with existing cache ─────────────────────────────────
    if cache is not None:
        if cached_df is None:
            cached_df = cache.load()

        frames = [f for f in (cached_df, new_df) if f is not None and not f.empty]
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            # Deduplicate on accession_number + transaction fields
            merged = merged.drop_duplicates(
                subset=[c for c in ('accession_number', 'transaction_date', 'shares',
                                    'reporting_owner', 'security_title')
                        if c in merged.columns],
                keep='last',
            )
        else:
            merged = pd.DataFrame()

        # Persist full merged data (all accessions ever parsed for this ticker)
        all_parsed = cached_accessions | new_accessions
        if not merged.empty:
            cache.save(_finalize_df(merged.copy()), all_parsed)

        # Return only the rows the caller asked for (date-filtered subset)
        if not merged.empty:
            result = merged[merged['accession_number'].isin(needed_accessions)]
        else:
            result = merged
        return _finalize_df(result, limit)

    # ── 5. No cache — return parsed data directly ────────────────────
    return _finalize_df(new_df, limit)
