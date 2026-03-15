"""
HTTP utilities for making requests with proper headers and error handling.
"""

import os
import certifi
import requests
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.constants import (
    DEFAULT_TIMEOUT,
    RETRY_COUNT,
    RETRY_BACKOFF_FACTOR,
    RETRY_STATUS_CODES,
    USER_AGENT,
    ACCEPT_HEADER,
    ACCEPT_LANGUAGE,
)


def _ssl_verify():
    """Use certifi CA bundle, or disable verify if SSL_VERIFY=false (e.g. macOS local dev)."""
    if os.getenv("SSL_VERIFY", "true").lower() in ("false", "0", "no"):
        return False
    return certifi.where()


def get_session_with_retries() -> requests.Session:
    """
    Create a requests session with retry strategy and SSL verify from env/certifi.
    
    Returns:
        requests.Session: Configured session with retry logic
    """
    session = requests.Session()
    session.verify = _ssl_verify()

    retry_strategy = Retry(
        total=RETRY_COUNT,
        backoff_factor=RETRY_BACKOFF_FACTOR,
        status_forcelist=RETRY_STATUS_CODES,
        allowed_methods=["HEAD", "GET", "OPTIONS"]
    )

    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session


def get_headers() -> Dict[str, str]:
    """
    Get standard headers for HTTP requests.
    
    Returns:
        Dict[str, str]: Headers dictionary
    """
    return {
        "User-Agent": USER_AGENT,
        "Accept": ACCEPT_HEADER,
        "Accept-Language": ACCEPT_LANGUAGE,
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1"
    }


def make_request(
    url: str,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = DEFAULT_TIMEOUT,
    **kwargs
) -> Optional[requests.Response]:
    """
    Make HTTP request with error handling.
    
    Args:
        url: Request URL
        method: HTTP method (GET, POST, etc.)
        params: Query parameters
        headers: Custom headers
        timeout: Request timeout in seconds
        **kwargs: Additional arguments for requests
    
    Returns:
        requests.Response: Response object or None if error
    
    Raises:
        requests.RequestException: If request fails
    """
    session = get_session_with_retries()
    
    if headers is None:
        headers = get_headers()
    
    try:
        response = session.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            timeout=timeout,
            verify=_ssl_verify(),
            **kwargs
        )
        response.raise_for_status()
        return response
    except requests.RequestException as e:
        raise requests.RequestException(f"Request failed for {url}: {str(e)}")
    finally:
        session.close()
