import logging
from typing import List, Dict, Any, Optional

from bs4 import BeautifulSoup

from models.stock_models import CompanyFundamentals, Fundamentals
from utils.http_utils import make_request
from utils.constants import (
    SCREENER_BASE_URL,
    SCREENER_TOP_RATIOS_ID,
    SCREENER_PROS_CLASS,
    SCREENER_CONS_CLASS,
    SCREENER_SECTION_QUARTERS,
    SCREENER_SECTION_PROFIT_LOSS,
    SCREENER_SECTION_BALANCE_SHEET,
    SCREENER_SECTION_CASH_FLOW,
    SCREENER_SECTION_RATIOS,
    SCREENER_SECTION_SHAREHOLDING,
    SCREENER_SECTION_PEERS,
    NOT_AVAILABLE_VALUES,
    METRIC_COLUMN_NAME,
    NUMBER_STRIP_CHARS,
)

logger = logging.getLogger(__name__)


def parse_number(value: Any) -> Optional[float]:
    """
    Parse a string value to a float, handling commas, percentages, and currency symbols.
    
    Args:
        value: String value like "125,849", "12%", "1,022.45 Cr."
    
    Returns:
        Float value or None if parsing fails
    """
    if value is None:
        return None
    
    if isinstance(value, (int, float)):
        return float(value)
    
    if not isinstance(value, str):
        return None
    
    cleaned = value.strip()
    if cleaned in NOT_AVAILABLE_VALUES:
        return None
    
    # Remove common formatting characters
    for char in NUMBER_STRIP_CHARS:
        cleaned = cleaned.replace(char, "")
    
    cleaned = cleaned.strip()
    if not cleaned:
        return None
    
    try:
        return float(cleaned)
    except ValueError:
        return None


class ScreenerProvider:
    BASE_URL = SCREENER_BASE_URL

    def _parse_financial_table(self, table) -> List[Dict[str, Any]]:
        """Turn a <table> element into a list of dicts with numeric values.

        The first row is treated as headers; subsequent rows are zipped against
        those headers. Empty header names are renamed to "metric".
        String values are converted to numbers where possible.
        """
        try:
            headers: List[str] = []
            rows = table.find_all("tr")
            if not rows:
                return []

            # header row - rename empty headers to "metric"
            for th in rows[0].find_all(["th", "td"]):
                header_text = th.get_text(strip=True)
                if not header_text:
                    header_text = METRIC_COLUMN_NAME
                headers.append(header_text)

            data: List[Dict[str, Any]] = []
            for row in rows[1:]:
                cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if not cells:
                    continue
                row_dict: Dict[str, Any] = {}
                for idx, header in enumerate(headers):
                    raw_value = cells[idx] if idx < len(cells) else ""
                    # Keep metric column as string, convert others to numbers
                    if header == METRIC_COLUMN_NAME:
                        # Clean up metric name (remove trailing +, -, etc.)
                        row_dict[header] = raw_value.rstrip("+-")
                    else:
                        # Try to convert to number, fallback to string
                        numeric_value = parse_number(raw_value)
                        row_dict[header] = numeric_value if numeric_value is not None else raw_value
                data.append(row_dict)
            return data
        except Exception as e:  # pragma: no cover - defensive
            logger.warning("failed to parse financial table: %s", e)
            return []

    def _parse_section_table(self, soup: BeautifulSoup, section_title: str) -> List[Dict[str, Any]]:
        """Find a heading containing *section_title* and parse the following table.

        Comparison is case‑insensitive; if the heading or table is missing an empty
        list is returned.
        """
        try:
            heading = soup.find(
                lambda tag: tag.name in ["h1", "h2", "h3", "h4", "h5", "h6"]
                and section_title.lower() in tag.get_text(strip=True).lower()
            )
            if not heading:
                return []

            table = heading.find_next("table")
            if not table:
                return []
            return self._parse_financial_table(table)
        except Exception as e:  # pragma: no cover
            logger.warning("error parsing section '%s': %s", section_title, e)
            return []

    def _parse_float(self, value: str) -> Optional[float]:
        """Parse a string to float, handling commas, percentages, and missing values."""
        if not value or value.strip() in NOT_AVAILABLE_VALUES:
            return None
        try:
            cleaned = value.replace(",", "").replace("%", "").strip()
            return float(cleaned)
        except ValueError:
            return None

    def get_fundamentals(self, symbol: str) -> Fundamentals:
        """Retrieve key fundamentals for a company symbol from Screener.in."""
        try:
            # Screener.in uses /company/ not /companies/
            url = f"{self.BASE_URL}/company/{symbol}/"
            logger.info(f"Fetching fundamentals from: {url}")
            response = make_request(url)
            if response is None:
                logger.warning(f"No response from Screener.in for {symbol}")
                return Fundamentals(symbol=symbol)

            content = getattr(response, "content", None)
            html_text = ""
            if isinstance(content, (bytes, bytearray)):
                html_text = content.decode("utf-8", errors="ignore")
            else:
                html_text = getattr(response, "text", "") or ""

            soup = BeautifulSoup(html_text, "html.parser")

            # Initialize fundamentals with symbol
            fundamentals = Fundamentals(symbol=symbol)
            
            # Parse top-ratios section
            top_ratios = soup.find("ul", id=SCREENER_TOP_RATIOS_ID)
            if top_ratios:
                for li in top_ratios.find_all("li"):
                    name_span = li.find("span", class_="name")
                    value_span = li.find("span", class_="number")
                    
                    if not name_span:
                        continue
                        
                    key = name_span.get_text(strip=True).lower()
                    val = value_span.get_text(strip=True) if value_span else None
                    
                    if not val:
                        continue
                    
                    # Map each ratio to the appropriate field
                    if "market cap" in key:
                        fundamentals.market_cap = self._parse_float(val)
                    elif key == "current price":
                        fundamentals.current_price = self._parse_float(val)
                    elif "high / low" in key or "high/low" in key:
                        # Store as string since it's a range
                        fundamentals.high_low = val
                    elif "stock p/e" in key or key in ("pe", "pe ratio", "p/e"):
                        fundamentals.pe_ratio = self._parse_float(val)
                    elif key == "book value":
                        fundamentals.book_value = self._parse_float(val)
                    elif "dividend yield" in key:
                        fundamentals.dividend_yield = self._parse_float(val)
                    elif key == "roce":
                        fundamentals.roce = self._parse_float(val)
                    elif key == "roe":
                        fundamentals.roe = self._parse_float(val)
                    elif key == "face value":
                        fundamentals.face_value = self._parse_float(val)
                    elif key == "eps":
                        fundamentals.eps = self._parse_float(val)
                    elif "debt" in key:
                        fundamentals.debt_equity = self._parse_float(val)
            else:
                logger.warning(f"No #top-ratios found for {symbol}")

            # Parse pros and cons
            pros_section = soup.find("div", class_=SCREENER_PROS_CLASS)
            cons_section = soup.find("div", class_=SCREENER_CONS_CLASS)
            
            if pros_section:
                fundamentals.pros = [li.get_text(strip=True) for li in pros_section.find_all("li")]
            if cons_section:
                fundamentals.cons = [li.get_text(strip=True) for li in cons_section.find_all("li")]

            logger.info(f"Parsed fundamentals for {symbol}: PE={fundamentals.pe_ratio}, ROE={fundamentals.roe}, ROCE={fundamentals.roce}, MarketCap={fundamentals.market_cap}")
            return fundamentals
        except Exception as e:
            logger.warning(f"Failed to get fundamentals for {symbol}: {str(e)}")
            return Fundamentals(symbol=symbol)
    
    def get_all_company_data(self, symbol: str) -> Dict[str, Any]:
        """
        Retrieve all available company data from Screener.in including financial tables.
        
        Returns a dict with:
        - fundamentals: Fundamentals object with top-ratios and pros/cons
        - quarterly_results: List of quarterly result rows
        - profit_loss: Annual P&L statement rows
        - balance_sheet: Balance sheet rows
        - cash_flow: Cash flow statement rows
        - ratios: Financial ratios over time
        - shareholding: Shareholding pattern over time
        - peer_comparison: Peer company comparison (if available)
        """
        try:
            url = f"{self.BASE_URL}/company/{symbol}/"
            logger.info(f"Fetching all company data from: {url}")
            response = make_request(url)
            
            if response is None:
                logger.warning(f"No response from Screener.in for {symbol}")
                return {"fundamentals": Fundamentals(symbol=symbol)}
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Get fundamentals first (reuse the parsing logic)
            fundamentals = self._parse_fundamentals_from_soup(soup, symbol)
            
            # Parse all financial tables by section ID
            quarterly_results = self._parse_section_by_id(soup, SCREENER_SECTION_QUARTERS)
            profit_loss = self._parse_section_by_id(soup, SCREENER_SECTION_PROFIT_LOSS)
            balance_sheet = self._parse_section_by_id(soup, SCREENER_SECTION_BALANCE_SHEET)
            cash_flow = self._parse_section_by_id(soup, SCREENER_SECTION_CASH_FLOW)
            ratios = self._parse_section_by_id(soup, SCREENER_SECTION_RATIOS)
            shareholding = self._parse_section_by_id(soup, SCREENER_SECTION_SHAREHOLDING)
            peer_comparison = self._parse_section_by_id(soup, SCREENER_SECTION_PEERS)
            
            return {
                "fundamentals": fundamentals,
                "quarterly_results": quarterly_results,
                "profit_loss": profit_loss,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
                "ratios": ratios,
                "shareholding": shareholding,
                "peer_comparison": peer_comparison,
            }
        except Exception as e:
            logger.error(f"Failed to get all company data for {symbol}: {str(e)}")
            return {"fundamentals": Fundamentals(symbol=symbol)}
    
    def _parse_fundamentals_from_soup(self, soup: BeautifulSoup, symbol: str) -> Fundamentals:
        """Parse fundamentals from an already-fetched BeautifulSoup object."""
        fundamentals = Fundamentals(symbol=symbol)
        
        top_ratios = soup.find("ul", id=SCREENER_TOP_RATIOS_ID)
        if top_ratios:
            for li in top_ratios.find_all("li"):
                name_span = li.find("span", class_="name")
                value_span = li.find("span", class_="number")
                
                if not name_span:
                    continue
                    
                key = name_span.get_text(strip=True).lower()
                val = value_span.get_text(strip=True) if value_span else None
                
                if not val:
                    continue
                
                if "market cap" in key:
                    fundamentals.market_cap = self._parse_float(val)
                elif key == "current price":
                    fundamentals.current_price = self._parse_float(val)
                elif "high / low" in key or "high/low" in key:
                    fundamentals.high_low = val
                elif "stock p/e" in key or key in ("pe", "pe ratio", "p/e"):
                    fundamentals.pe_ratio = self._parse_float(val)
                elif key == "book value":
                    fundamentals.book_value = self._parse_float(val)
                elif "dividend yield" in key:
                    fundamentals.dividend_yield = self._parse_float(val)
                elif key == "roce":
                    fundamentals.roce = self._parse_float(val)
                elif key == "roe":
                    fundamentals.roe = self._parse_float(val)
                elif key == "face value":
                    fundamentals.face_value = self._parse_float(val)
                elif key == "eps":
                    fundamentals.eps = self._parse_float(val)
                elif "debt" in key:
                    fundamentals.debt_equity = self._parse_float(val)
        
        # Parse pros and cons
        pros_section = soup.find("div", class_="pros")
        cons_section = soup.find("div", class_="cons")
        
        if pros_section:
            fundamentals.pros = [li.get_text(strip=True) for li in pros_section.find_all("li")]
        if cons_section:
            fundamentals.cons = [li.get_text(strip=True) for li in cons_section.find_all("li")]
        
        return fundamentals
    
    def _parse_section_by_id(self, soup: BeautifulSoup, section_id: str) -> List[Dict[str, Any]]:
        """Parse a table from a section identified by its ID."""
        try:
            section = soup.find("section", id=section_id)
            if not section:
                return []
            
            table = section.find("table")
            if not table:
                return []
            
            return self._parse_financial_table(table)
        except Exception as e:
            logger.warning(f"Error parsing section '{section_id}': {str(e)}")
            return []

    def get_company_data(self, symbol: str) -> CompanyFundamentals:
        """Retrieve all company data for *symbol* from Screener.in.

        Returns a :class:`CompanyFundamentals` instance; callers who prefer a plain
        ``dict`` can convert it with ``dataclasses.asdict``.
        """
        url = f"{self.BASE_URL}/companies/{symbol}/"
        response = make_request(url)
        if response.status_code != 200:
            logger.error("failed to fetch %s: status %s", url, response.status_code)
            return CompanyFundamentals(company_info={"symbol": symbol})

        soup = BeautifulSoup(response.text, "html.parser")

        company_info: Dict[str, Any] = {"symbol": symbol}
        title_tag = soup.find("h1")
        if title_tag:
            company_info["name"] = title_tag.get_text(strip=True)

        # fundamentals block
        fundamentals: Dict[str, Any] = {}
        for li in soup.select("li.flex.flex-space-between"):
            name_span = li.find("span", class_="name")
            data_span = li.find("span", class_="data")
            if not name_span or not data_span:
                continue
            fundamentals[name_span.get_text(strip=True)] = data_span.get_text(strip=True)

        pros = [p.get_text(strip=True) for p in soup.select("div.procon div.pro > p")]
        cons = [p.get_text(strip=True) for p in soup.select("div.procon div.con > p")]
        pros_cons = {"pros": pros, "cons": cons}

        peer_comparison = self._parse_section_table(soup, "Peer comparison")
        quarterly_results = self._parse_section_table(soup, "Quarterly Results")
        profit_loss = self._parse_section_table(soup, "Profit & Loss")
        balance_sheet = self._parse_section_table(soup, "Balance Sheet")
        cash_flow = self._parse_section_table(soup, "Cash Flow")
        ratios = self._parse_section_table(soup, "Ratios")
        shareholding = self._parse_section_table(soup, "Shareholding Pattern")

        
        return CompanyFundamentals(
            company_info=company_info,
            fundamentals=fundamentals,
            pros_cons=pros_cons,
            peer_comparison=peer_comparison,
            quarterly_results=quarterly_results,
            profit_loss=profit_loss,
            balance_sheet=balance_sheet,
            cash_flow=cash_flow,
            ratios=ratios,
            shareholding=shareholding,
        )
