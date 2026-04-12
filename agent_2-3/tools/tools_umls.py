# -*- coding: utf-8 -*-
"""
UMLS (Unified Medical Language System) API Tools.
This module provides functions to interact with UMLS API for medical terminology standardization.
"""
import os
import time
import re
from typing import Dict, Optional, List, Any, Tuple

# Try to import requests
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("Warning: requests library not found. UMLS API functionality will be disabled.")
    print("Install it with: pip install requests")

# UMLS API endpoints
UMLS_BASE_URL = "https://uts-ws.nlm.nih.gov/rest"
# UMLS authentication uses CAS (Central Authentication Service)
# First get TGT from login service, then get ST from ticket service
UMLS_LOGIN_URL = "https://utslogin.nlm.nih.gov/cas/v1/api-key"  # For API key authentication
UMLS_TICKET_URL = "https://utslogin.nlm.nih.gov/cas/v1/tickets"  # For getting service ticket
UMLS_AUTH_URL = f"{UMLS_BASE_URL}/authentication/v1/ticket"  # Legacy endpoint (may not work)
UMLS_SEARCH_URL = f"{UMLS_BASE_URL}/search/current"
UMLS_CONTENT_URL = f"{UMLS_BASE_URL}/content/current"

# Cache for UMLS tickets (valid for 8 hours)
_UMLS_TICKET: Optional[str] = None
_TICKET_EXPIRY: float = 0

# Cache for authentication errors to avoid repeated attempts
_UMLS_AUTH_ERROR: Optional[str] = None
_UMLS_AUTH_ERROR_TIME: float = 0


def get_umls_credentials() -> tuple:
    """
    Get UMLS credentials from environment variables.
    
    Returns:
        Tuple of (api_key, username, password) or (None, None, None) if not set
    """
    # UMLS can use either API key or username/password
    api_key = os.environ.get("UMLS_API_KEY")
    username = os.environ.get("UMLS_USERNAME")
    password = os.environ.get("UMLS_PASSWORD")
    
    # Clean API key (remove whitespace)
    if api_key:
        api_key = api_key.strip()
        # Validate API key format (UMLS API keys are typically UUIDs)
        if not api_key or len(api_key) < 10:
            print("警告: UMLS_API_KEY 格式可能不正确（长度过短）")
    
    return api_key, username, password


def _get_requests_session():
    """
    Get a requests session with proper proxy configuration.
    
    Returns:
        requests.Session configured with proxy settings
    """
    session = requests.Session()
    
    # Check if we should disable proxy for UMLS
    disable_proxy = os.environ.get("UMLS_DISABLE_PROXY", "false").lower() == "true"
    
    if disable_proxy:
        # Explicitly disable proxy
        session.proxies = {
            "http": None,
            "https": None
        }
    else:
        # Use environment proxy settings if available
        # requests library automatically uses HTTP_PROXY and HTTPS_PROXY
        pass
    
    return session


def get_umls_ticket(force_refresh: bool = False) -> Optional[str]:
    """
    Get a valid UMLS ticket for API authentication.
    Tickets are valid for 8 hours and are cached.
    
    Note: UMLS Service Tickets (ST) are single-use. This function caches the ST
    but will refresh it automatically when a 401 error is detected.
    
    Args:
        force_refresh: Force refresh of the ticket even if cached one is still valid
    
    Returns:
        UMLS ticket string or None if authentication fails
    """
    if not REQUESTS_AVAILABLE:
        return None
    
    global _UMLS_TICKET, _TICKET_EXPIRY, _UMLS_AUTH_ERROR, _UMLS_AUTH_ERROR_TIME
    
    # Return cached ticket if still valid
    if not force_refresh and _UMLS_TICKET and time.time() < _TICKET_EXPIRY:
        return _UMLS_TICKET
    
    # If we recently failed authentication, don't retry immediately
    # Wait at least 5 minutes before retrying
    if not force_refresh and _UMLS_AUTH_ERROR and time.time() - _UMLS_AUTH_ERROR_TIME < 300:
        verbose = os.environ.get("UMLS_VERBOSE", "false").lower() == "true"
        if verbose:
            print(f"[UMLS] 跳过认证重试（上次失败: {_UMLS_AUTH_ERROR}）")
        return None
    
    # Clear previous error
    _UMLS_AUTH_ERROR = None
    
    api_key, username, password = get_umls_credentials()
    
    # Check if credentials are available
    if not api_key and not (username and password):
        error_msg = "UMLS credentials not found. Please set UMLS_API_KEY or UMLS_USERNAME/UMLS_PASSWORD environment variables."
        _UMLS_AUTH_ERROR = error_msg
        _UMLS_AUTH_ERROR_TIME = time.time()
        print(f"[UMLS] 错误: {error_msg}")
        print(f"[UMLS] 当前环境变量检查:")
        print(f"  - UMLS_API_KEY: {'已设置' if os.environ.get('UMLS_API_KEY') else '未设置'}")
        print(f"  - UMLS_USERNAME: {'已设置' if os.environ.get('UMLS_USERNAME') else '未设置'}")
        print(f"  - UMLS_PASSWORD: {'已设置' if os.environ.get('UMLS_PASSWORD') else '未设置'}")
        return None
    
    # Create session with proxy configuration
    session = _get_requests_session()
    
    # Try API key first (preferred method)
    if api_key:
        verbose = os.environ.get("UMLS_VERBOSE", "false").lower() == "true"
        if verbose:
            print(f"[UMLS] 尝试使用API key认证 (key长度: {len(api_key)})")
        
        # Try new CAS authentication method first
        try:
            if verbose:
                print(f"[UMLS] 尝试CAS认证 (URL: {UMLS_LOGIN_URL})")
            response = session.post(
                UMLS_LOGIN_URL,
                data={"apikey": api_key},
                timeout=10
            )
            
            if response.status_code == 201:  # CAS returns 201 for successful TGT creation
                # Extract TGT from Location header
                location = response.headers.get("Location", "")
                if location:
                    tgt = location.split("/")[-1]
                    if verbose:
                        print(f"[UMLS] 成功获取TGT: {tgt[:20]}...")
                else:
                    # Try to get TGT from response text
                    tgt = response.text.strip()
                    if tgt:
                        if verbose:
                            print(f"[UMLS] 成功获取TGT (从响应文本)")
                    else:
                        raise Exception("无法从响应中提取TGT")
            elif response.status_code == 400:
                # 400 Bad Request usually means invalid API key
                error_text = response.text[:500]
                raise Exception(f"CAS认证失败 (400 Bad Request): API key可能无效。响应: {error_text}")
            else:
                # If CAS method fails, try legacy method
                error_text = response.text[:500] if response.text else "无响应内容"
                if verbose:
                    print(f"[UMLS] CAS认证返回状态码 {response.status_code}，响应: {error_text}")
                raise Exception(f"CAS认证失败，状态码: {response.status_code}, 响应: {error_text}")
            
            # Get service ticket using TGT
            # The service URL should match the actual API endpoint we'll use
            service_url = f"{UMLS_TICKET_URL}/{tgt}"
            # Use the REST API service URL - must match exactly what UMLS expects
            # Try different service URLs if one doesn't work
            service_urls_to_try = [
                "http://umlsks.nlm.nih.gov",
                "https://uts-ws.nlm.nih.gov",
                "http://umlsks.nlm.nih.gov/restful"
            ]
            
            service_ticket = None
            for service_url_param in service_urls_to_try:
                service_data = {"service": service_url_param}
                
                if verbose:
                    print(f"[UMLS] 获取服务票证 (ST)，尝试服务URL: {service_url_param}")
                service_response = session.post(service_url, data=service_data, timeout=10)
                
                if verbose:
                    print(f"[UMLS] 服务票证响应状态码: {service_response.status_code}")
                    if service_response.status_code != 200:
                        print(f"[UMLS] 服务票证响应内容: {service_response.text[:200]}")
                
                if service_response.status_code == 200:
                    service_ticket = service_response.text.strip()
                    if service_ticket:
                        if verbose:
                            print(f"[UMLS] ✓ 成功获取服务票证 (使用服务URL: {service_url_param})")
                        break
            
            if service_ticket:
                _UMLS_TICKET = service_ticket
                _TICKET_EXPIRY = time.time() + 8 * 3600 - 300  # 8 hours minus 5 min buffer
                if verbose:
                    print("[UMLS] ✓ 认证成功，已获取服务票证")
                return _UMLS_TICKET
            else:
                raise Exception("无法获取服务票证，所有服务URL都失败")
                
        except requests.exceptions.ProxyError as e:
            # Cache proxy errors to avoid repeated attempts
            _UMLS_AUTH_ERROR = f"Proxy error: {str(e)}"
            _UMLS_AUTH_ERROR_TIME = time.time()
            if verbose:
                print(f"UMLS API authentication failed (proxy error): {e}")
                print("提示: 如果您的网络使用代理，可以设置环境变量 UMLS_DISABLE_PROXY=true 来禁用代理")
            return None
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (like 400 Bad Request)
            error_msg = str(e)
            _UMLS_AUTH_ERROR = error_msg
            _UMLS_AUTH_ERROR_TIME = time.time()
            if verbose:
                print(f"[UMLS] API key认证失败: {e}")
            return None
        except Exception as cas_error:
            # CAS认证失败，不再尝试legacy方法（已废弃）
            error_msg = f"CAS authentication failed: {str(cas_error)}"
            _UMLS_AUTH_ERROR = error_msg
            _UMLS_AUTH_ERROR_TIME = time.time()
            if verbose:
                print(f"[UMLS] CAS认证失败: {cas_error}")
                print(f"[UMLS] 提示: 请检查UMLS_API_KEY是否正确，或网络连接是否正常")
                print("  1. API key 无效或已过期 - 请检查您的 UMLS API key 是否正确")
                print("  2. API key 格式错误 - 请确保 API key 没有多余的空格或换行符")
                print("  3. 网络连接问题 - 请检查网络连接或尝试禁用代理 (UMLS_DISABLE_PROXY=true)")
                print("  4. UMLS 服务暂时不可用 - 请稍后重试")
                print(f"[UMLS] 提示: 如果问题持续，请访问 https://uts.nlm.nih.gov/uts/ 验证您的API key")
            return None
        except Exception as e:
            # Cache other errors too
            error_msg = str(e)
            if "proxy" in error_msg.lower() or "ProxyError" in str(type(e)):
                _UMLS_AUTH_ERROR = f"Proxy error: {error_msg}"
            else:
                _UMLS_AUTH_ERROR = error_msg
            _UMLS_AUTH_ERROR_TIME = time.time()
            # Only print error once
            if force_refresh or not _UMLS_AUTH_ERROR_TIME or time.time() - _UMLS_AUTH_ERROR_TIME > 300:
                print(f"UMLS API key authentication failed: {e}")
                # Print detailed error information for debugging
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        print(f"  错误详情: {error_detail}")
                    except:
                        if hasattr(e.response, 'status_code'):
                            print(f"  响应状态码: {e.response.status_code}")
                        if hasattr(e.response, 'text'):
                            print(f"  响应内容: {e.response.text[:500]}")
    
    # Fall back to username/password
    if username and password:
        try:
            response = session.post(
                UMLS_AUTH_URL,
                data={"username": username, "password": password},
                timeout=10
            )
            response.raise_for_status()
            result = response.json()
            ticket = result.get("result", {}).get("tgt")
            
            if ticket:
                # Get service ticket
                service_response = session.post(
                    f"{UMLS_AUTH_URL}/{ticket}",
                    data={"service": "http://umlsks.nlm.nih.gov"},
                    timeout=10
                )
                service_response.raise_for_status()
                service_result = service_response.json()
                service_ticket = service_result.get("result", {}).get("st")
                
                if service_ticket:
                    _UMLS_TICKET = service_ticket
                    _TICKET_EXPIRY = time.time() + 8 * 3600 - 300
                    verbose = os.environ.get("UMLS_VERBOSE", "false").lower() == "true"
                    if verbose:
                        print("[UMLS] ✓ 认证成功，已获取服务票证")
                    return _UMLS_TICKET
        except requests.exceptions.ProxyError as e:
            _UMLS_AUTH_ERROR = f"Proxy error: {str(e)}"
            _UMLS_AUTH_ERROR_TIME = time.time()
            if force_refresh or not _UMLS_AUTH_ERROR_TIME or time.time() - _UMLS_AUTH_ERROR_TIME > 300:
                print(f"UMLS API authentication failed (proxy error): {e}")
                print("提示: 如果您的网络使用代理，可以设置环境变量 UMLS_DISABLE_PROXY=true 来禁用代理")
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (like 400 Bad Request)
            error_msg = str(e)
            _UMLS_AUTH_ERROR = error_msg
            _UMLS_AUTH_ERROR_TIME = time.time()
            if force_refresh or not _UMLS_AUTH_ERROR_TIME or time.time() - _UMLS_AUTH_ERROR_TIME > 300:
                print(f"UMLS username/password authentication failed: {e}")
                # Print detailed error information
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        print(f"  错误详情: {error_detail}")
                    except:
                        print(f"  响应状态码: {e.response.status_code}")
                        print(f"  响应内容: {e.response.text[:500]}")
        except Exception as e:
            error_msg = str(e)
            if "proxy" in error_msg.lower() or "ProxyError" in str(type(e)):
                _UMLS_AUTH_ERROR = f"Proxy error: {error_msg}"
            else:
                _UMLS_AUTH_ERROR = error_msg
            _UMLS_AUTH_ERROR_TIME = time.time()
            if force_refresh or not _UMLS_AUTH_ERROR_TIME or time.time() - _UMLS_AUTH_ERROR_TIME > 300:
                print(f"UMLS username/password authentication failed: {e}")
                # Print detailed error information for debugging
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        print(f"  错误详情: {error_detail}")
                    except:
                        if hasattr(e.response, 'status_code'):
                            print(f"  响应状态码: {e.response.status_code}")
                        if hasattr(e.response, 'text'):
                            print(f"  响应内容: {e.response.text[:500]}")
                # Print detailed error information for debugging
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_detail = e.response.json()
                        print(f"  错误详情: {error_detail}")
                    except:
                        print(f"  响应状态码: {e.response.status_code}")
                        print(f"  响应内容: {e.response.text[:500]}")
    
    return None


def search_umls(
    term: str,
    page_size: int = 25,
    page_number: int = 1,
    search_type: str = "exact",
    sabs: Optional[str] = None,
    language: str = "ENG"
) -> Dict[str, Any]:
    """
    Search UMLS for a medical term.
    
    Args:
        term: The medical term to search for
        page_size: Number of results per page (default 25, max 25)
        page_number: Page number (default 1)
        search_type: Search type - "exact", "words", "leftTruncation", "rightTruncation", "approximate"
        sabs: Source vocabulary filter (e.g., "SNOMEDCT_US,ICD10CM,LOINC")
        language: Language filter (default "ENG" for English)
    
    Returns:
        Dictionary containing search results with 'result' key containing list of concepts
    """
    if not REQUESTS_AVAILABLE:
        return {"error": "requests library not available. Install with: pip install requests"}
    
    ticket = get_umls_ticket()
    if not ticket:
        # Check if there's a cached error
        global _UMLS_AUTH_ERROR, _UMLS_AUTH_ERROR_TIME
        if _UMLS_AUTH_ERROR:
            error_msg = f"UMLS authentication failed: {_UMLS_AUTH_ERROR}"
            # If error is recent, suggest clearing cache
            if time.time() - _UMLS_AUTH_ERROR_TIME < 300:
                error_msg += f" (错误已缓存，可使用 clear_umls_cache() 清除缓存后重试)"
        else:
            error_msg = "UMLS authentication failed. Please set UMLS_API_KEY or UMLS_USERNAME/UMLS_PASSWORD environment variables."
        return {"error": error_msg}
    
    # Try with current ticket, retry once with new ticket if 401
    verbose = os.environ.get("UMLS_VERBOSE", "false").lower() == "true"
    for retry in range(2):
        try:
            session = _get_requests_session()
            params = {
                "string": term,
                "ticket": ticket,
                "pageSize": min(page_size, 25),
                "pageNumber": page_number,
                "searchType": search_type
            }
            # Add optional filters
            if sabs:
                params["sabs"] = sabs
            if language:
                params["language"] = language
            
            if verbose:
                print(f"[UMLS] 搜索参数: string={term}, searchType={search_type}, ticket={ticket[:20]}...")
                print(f"[UMLS] 搜索URL: {UMLS_SEARCH_URL}")
            
            response = session.get(UMLS_SEARCH_URL, params=params, timeout=10)
            
            if verbose:
                print(f"[UMLS] 响应状态码: {response.status_code}")
            
            # Check for 401 Unauthorized - ticket expired or invalid (ST is single-use)
            if response.status_code == 401:
                if retry == 0:
                    # Clear cached ticket and get a new one
                    if verbose:
                        print(f"[UMLS] 检测到401错误（服务票证已失效），刷新服务票证并重试...")
                    clear_umls_cache()
                    ticket = get_umls_ticket(force_refresh=True)
                    if not ticket:
                        return {"error": "UMLS authentication failed after ticket refresh"}
                    continue  # Retry with new ticket
                else:
                    # Already retried once, return error
                    error_msg = f"UMLS search failed: 401 Unauthorized (ticket refresh did not help)"
                    if verbose:
                        print(f"[UMLS] {error_msg}")
                    return {"error": error_msg}
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Check if it's a 401 error
            is_401 = hasattr(e, 'response') and e.response and e.response.status_code == 401
            if is_401 and retry == 0:
                # Clear cached ticket and get a new one
                if verbose:
                    print(f"[UMLS] 检测到401错误（服务票证已失效），刷新服务票证并重试...")
                clear_umls_cache()
                ticket = get_umls_ticket(force_refresh=True)
                if not ticket:
                    return {"error": "UMLS authentication failed after ticket refresh"}
                continue  # Retry with new ticket
            
            # For non-401 errors or after retry, return error
            error_msg = f"UMLS search failed: {str(e)}"
            if verbose and hasattr(e, 'response') and e.response is not None:
                print(f"[UMLS] 搜索错误详情:")
                print(f"[UMLS] 状态码: {e.response.status_code}")
                print(f"[UMLS] 响应头: {dict(e.response.headers)}")
                try:
                    error_detail = e.response.json()
                    print(f"[UMLS] 响应JSON: {error_detail}")
                except:
                    print(f"[UMLS] 响应文本: {e.response.text[:500]}")
            return {"error": error_msg}
        except Exception as e:
            return {"error": f"UMLS search failed: {str(e)}"}
    
    return {"error": "UMLS search failed after retry"}


def get_umls_concept(cui: str) -> Dict[str, Any]:
    """
    Get detailed information about a UMLS concept by CUI.
    
    Args:
        cui: Concept Unique Identifier (e.g., "C0004096")
    
    Returns:
        Dictionary containing concept details
    """
    if not REQUESTS_AVAILABLE:
        return {"error": "requests library not available"}
    
    ticket = get_umls_ticket()
    if not ticket:
        return {"error": "UMLS authentication failed"}
    
    # Try with current ticket, retry once with new ticket if 401
    for retry in range(2):
        try:
            session = _get_requests_session()
            url = f"{UMLS_CONTENT_URL}/CUI/{cui}"
            params = {"ticket": ticket}
            
            response = session.get(url, params=params, timeout=10)
            
            # Check for 401 Unauthorized - ticket expired or invalid (ST is single-use)
            if response.status_code == 401:
                if retry == 0:
                    # Clear cached ticket and get a new one
                    clear_umls_cache()
                    ticket = get_umls_ticket(force_refresh=True)
                    if not ticket:
                        return {"error": "UMLS authentication failed after ticket refresh"}
                    continue  # Retry with new ticket
                else:
                    # Already retried once, return error
                    return {"error": "UMLS concept retrieval failed: 401 Unauthorized (ticket refresh did not help)"}
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Check if it's a 401 error
            is_401 = hasattr(e, 'response') and e.response and e.response.status_code == 401
            if is_401 and retry == 0:
                # Clear cached ticket and get a new one
                clear_umls_cache()
                ticket = get_umls_ticket(force_refresh=True)
                if not ticket:
                    return {"error": "UMLS authentication failed after ticket refresh"}
                continue  # Retry with new ticket
            
            # For non-401 errors or after retry, return error
            return {"error": f"UMLS concept retrieval failed: {str(e)}"}
        except Exception as e:
            return {"error": f"UMLS concept retrieval failed: {str(e)}"}
    
    return {"error": "UMLS concept retrieval failed after retry"}


def get_umls_atoms(
    cui: str, 
    source: Optional[str] = None,
    language: str = "ENG"
) -> Dict[str, Any]:
    """
    Get atoms (terms) associated with a UMLS concept.
    
    Args:
        cui: Concept Unique Identifier
        source: Optional source vocabulary filter (e.g., "ICD10CM", "SNOMEDCT_US", "ATC")
        language: Language filter (default "ENG" for English)
    
    Returns:
        Dictionary containing atom information
    """
    if not REQUESTS_AVAILABLE:
        return {"error": "requests library not available"}
    
    ticket = get_umls_ticket()
    if not ticket:
        return {"error": "UMLS authentication failed"}
    
    # Try with current ticket, retry once with new ticket if 401
    for retry in range(2):
        try:
            session = _get_requests_session()
            url = f"{UMLS_CONTENT_URL}/CUI/{cui}/atoms"
            params = {"ticket": ticket}
            if source:
                params["sabs"] = source
            if language:
                params["language"] = language
            
            response = session.get(url, params=params, timeout=10)
            
            # Check for 401 Unauthorized - ticket expired or invalid (ST is single-use)
            if response.status_code == 401:
                if retry == 0:
                    # Clear cached ticket and get a new one
                    clear_umls_cache()
                    ticket = get_umls_ticket(force_refresh=True)
                    if not ticket:
                        return {"error": "UMLS authentication failed after ticket refresh"}
                    continue  # Retry with new ticket
                else:
                    # Already retried once, return error
                    return {"error": "UMLS atoms retrieval failed: 401 Unauthorized (ticket refresh did not help)"}
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Check if it's a 401 error
            is_401 = hasattr(e, 'response') and e.response and e.response.status_code == 401
            if is_401 and retry == 0:
                # Clear cached ticket and get a new one
                clear_umls_cache()
                ticket = get_umls_ticket(force_refresh=True)
                if not ticket:
                    return {"error": "UMLS authentication failed after ticket refresh"}
                continue  # Retry with new ticket
            
            # For non-401 errors or after retry, return error
            return {"error": f"UMLS atoms retrieval failed: {str(e)}"}
        except Exception as e:
            return {"error": f"UMLS atoms retrieval failed: {str(e)}"}
    
    return {"error": "UMLS atoms retrieval failed after retry"}


def standardize_term_with_umls(
    term: str,
    category: str,
    preferred_sources: Optional[List[str]] = None,
    verbose: bool = False,
    language: str = "ENG"
) -> Dict[str, str]:
    """
    Standardize a medical term using UMLS API.
    
    This function:
    1. Searches UMLS for the term
    2. Finds the best matching concept
    3. Extracts standard codes from preferred sources (ICD-10, SNOMED-CT, ATC, LOINC, etc.)
    
    Args:
        term: The medical term to standardize
        category: The category (Disease, Drug, Test, Symptom, etc.)
        preferred_sources: List of preferred source vocabularies (e.g., ["ICD10CM", "SNOMEDCT_US", "ATC"])
        verbose: Whether to print debug information
        language: Language for results (default "ENG" for English)
    
    Returns:
        Dictionary with 'code', 'system', 'name', 'cui', 'source' if found, empty dict otherwise
    """
    if not term or not term.strip():
        return {}
    
    if verbose:
        print(f"[UMLS] 正在查询: {term} ({category}), 语言: {language}")
    
    # Map category to preferred UMLS sources (English sources only)
    # Priority order: try primary source first, then fallbacks
    if preferred_sources is None:
        category_source_map = {
            "Disease": ["ICD10CM", "SNOMEDCT_US", "ICD10", "MSH"],
            "Drug": ["ATC", "RXNORM", "NDC", "MSH"],
            "Test": ["LOINC", "SNOMEDCT_US", "MSH"],
            "Symptom": ["SNOMEDCT_US", "ICD10CM", "MSH"],
            "Treatment": ["SNOMEDCT_US", "ICD10CM", "MSH"],
            "Anatomy": ["SNOMEDCT_US", "MSH"],
            "LabValue": ["LOINC", "SNOMEDCT_US"]
        }
        preferred_sources = category_source_map.get(category, ["SNOMEDCT_US", "MSH"])
    
    # Build source filter string for search (use primary source first)
    primary_source = preferred_sources[0] if preferred_sources else None
    sabs_filter = ",".join(preferred_sources) if preferred_sources else None
    
    # Search UMLS with language and source filters
    if verbose:
        print(f"[UMLS] 搜索术语: {term} (精确匹配, 语言={language}, 源={primary_source})")
    
    # First try searching in preferred source only
    search_result = search_umls(term, search_type="exact", sabs=primary_source, language=language)
    
    if "error" in search_result or not search_result.get("result", {}).get("results"):
        if verbose:
            print(f"[UMLS] 在{primary_source}中未找到，尝试其他源...")
        # Try with all preferred sources
        search_result = search_umls(term, search_type="exact", sabs=sabs_filter, language=language)
    
    if "error" in search_result or not search_result.get("result", {}).get("results"):
        if verbose:
            print(f"[UMLS] 精确搜索失败，尝试模糊搜索...")
        # Try approximate search without source filter
        search_result = search_umls(term, search_type="words", language=language)
    
    if "error" in search_result:
        error_msg = search_result.get("error", "Unknown error")
        if verbose:
            print(f"[UMLS] 模糊搜索也失败: {error_msg}")
        # Always print authentication errors even if not verbose
        if "authentication" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            print(f"[UMLS] 认证错误: {error_msg}")
            print(f"[UMLS] 请检查 UMLS_API_KEY 是否有效")
        return {}
    
    # Get results
    results = search_result.get("result", {}).get("results", [])
    if not results:
        if verbose:
            print(f"[UMLS] 未找到匹配结果 (搜索词: {term})")
        return {}
    
    if verbose:
        print(f"[UMLS] 找到 {len(results)} 个匹配结果")
    
    # Find best match (prefer exact matches, then by rank)
    best_match = None
    for result in results:
        if result.get("name", "").lower() == term.lower():
            best_match = result
            break
    
    if not best_match:
        best_match = results[0]  # Use top result
    
    cui = best_match.get("ui")
    if not cui:
        if verbose:
            print(f"[UMLS] 未找到CUI")
        return {}
    
    if verbose:
        print(f"[UMLS] 最佳匹配: {best_match.get('name', term)} (CUI: {cui})")
    
    # Get atoms to find codes from preferred sources (English only)
    if verbose:
        print(f"[UMLS] 获取概念详情 (CUI: {cui}, 语言: {language})")
    atoms_result = get_umls_atoms(cui, language=language)
    if "error" in atoms_result:
        if verbose:
            print(f"[UMLS] 获取atoms失败，使用CUI作为代码: {atoms_result.get('error', 'Unknown error')}")
        # Fallback: use concept name and CUI
        result = {
            "code": cui,
            "system": "UMLS",
            "name": best_match.get("name", term),
            "cui": cui,
            "source": "umls"
        }
        if verbose:
            print(f"[UMLS] ✓ 标准化成功: {term} -> {result['system']} {result['code']}")
        return result
    
    atoms = atoms_result.get("result", [])
    if verbose:
        print(f"[UMLS] 找到 {len(atoms)} 个atoms")
        # Show available sources for debugging
        available_sources = set(atom.get("rootSource", "") for atom in atoms)
        print(f"[UMLS] 可用的编码系统: {available_sources}")
    
    # Filter atoms to ensure English and valid sources
    # Exclude non-English sources (like MDRARA which is Arabic)
    non_english_sources = {"MDRARA", "MDRSPA", "MDRFRE", "MDRGER", "MDRDUT", "MDRJPN", "MDRCZE", "MDRHUN", "MDRPOR"}
    filtered_atoms = [
        atom for atom in atoms 
        if atom.get("rootSource", "") not in non_english_sources
        and atom.get("language", "ENG") == "ENG"
    ]
    
    if verbose and len(filtered_atoms) != len(atoms):
        print(f"[UMLS] 过滤非英语后剩余 {len(filtered_atoms)} 个atoms")
    
    # Find atom from preferred source (in priority order)
    for source in preferred_sources:
        for atom in filtered_atoms:
            atom_source = atom.get("rootSource", "")
            # Exact match first
            if atom_source == source:
                code_value = atom.get("code", cui)
                # Extract actual code from URL if needed
                if isinstance(code_value, str) and "/" in code_value:
                    code_value = code_value.split("/")[-1]
                result = {
                    "code": code_value,
                    "system": source,
                    "name": atom.get("name", best_match.get("name", term)),
                    "cui": cui,
                    "source": "umls"
                }
                if verbose:
                    print(f"[UMLS] ✓ 标准化成功: {term} -> {result['system']} {result['code']} ({result['name']})")
                return result
    
        # Partial match (e.g., "ICD10CM" matches "ICD10")
        for atom in filtered_atoms:
            atom_source = atom.get("rootSource", "")
            if source in atom_source or atom_source in source:
                code_value = atom.get("code", cui)
                if isinstance(code_value, str) and "/" in code_value:
                    code_value = code_value.split("/")[-1]
                result = {
                    "code": code_value,
                    "system": atom_source,
                    "name": atom.get("name", best_match.get("name", term)),
                    "cui": cui,
                    "source": "umls"
                }
                if verbose:
                    print(f"[UMLS] ✓ 标准化成功: {term} -> {result['system']} {result['code']} ({result['name']})")
                return result
    
    # If no preferred source found, use first English atom from allowed sources
    allowed_sources = {"SNOMEDCT_US", "ICD10CM", "ICD10", "LOINC", "ATC", "RXNORM", "MSH", "NCI"}
    for atom in filtered_atoms:
        atom_source = atom.get("rootSource", "")
        if atom_source in allowed_sources:
            code_value = atom.get("code", cui)
            if isinstance(code_value, str) and "/" in code_value:
                code_value = code_value.split("/")[-1]
            result = {
                "code": code_value,
                "system": atom_source,
                "name": atom.get("name", best_match.get("name", term)),
                "cui": cui,
                "source": "umls"
            }
            if verbose:
                print(f"[UMLS] ✓ 标准化成功(备选): {term} -> {result['system']} {result['code']} ({result['name']})")
            return result
    
    # Fallback to CUI with English name
    result = {
        "code": cui,
        "system": "UMLS",
        "name": best_match.get("name", term),
        "cui": cui,
        "source": "umls"
    }
    if verbose:
        print(f"[UMLS] ✓ 标准化成功(CUI): {term} -> {result['system']} {result['code']} ({result['name']})")
    return result


def get_loinc_unit_info(loinc_code: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Get unit information for a LOINC code from UMLS.
    
    This function queries UMLS to get the standard unit for a LOINC code.
    LOINC codes contain unit information in their atom names (e.g., "[Mass/volume]").
    
    Args:
        loinc_code: LOINC code (e.g., "2339-0", "17819-4")
        verbose: Whether to print debug information
    
    Returns:
        Dictionary containing unit information with keys:
        - standard_unit: Standard unit from LOINC (e.g., "mg/dL", "mmol/L")
        - unit_type: Type of unit (e.g., "Mass/volume", "#/volume")
        - error: Error message if query failed
    """
    if not REQUESTS_AVAILABLE:
        return {"error": "requests library not available"}
    
    if not loinc_code or not loinc_code.strip():
        return {"error": "LOINC code is required"}
    
    loinc_code = loinc_code.strip()
    
    if verbose:
        print(f"[LOINC] 查询LOINC编码单位信息: {loinc_code}")
    
    # Method 1: Direct access to LOINC source content via UMLS Content API
    # URL format: https://uts-ws.nlm.nih.gov/rest/content/current/source/LNC/{loinc_code}
    ticket = get_umls_ticket()
    if not ticket:
        return {"error": "UMLS authentication failed"}
    
    try:
        session = _get_requests_session()
        loinc_url = f"{UMLS_CONTENT_URL}/source/LNC/{loinc_code}"
        params = {"ticket": ticket}
        
        if verbose:
            print(f"[LOINC] 直接访问LOINC内容: {loinc_url}")
        
        response = session.get(loinc_url, params=params, timeout=10)
        
        if response.status_code == 404:
            if verbose:
                print(f"[LOINC] LOINC编码不存在: {loinc_code}")
            return {"error": f"LOINC code not found: {loinc_code}"}
        
        # Handle 401 - get new ticket and retry
        if response.status_code == 401:
            clear_umls_cache()
            ticket = get_umls_ticket(force_refresh=True)
            if ticket:
                params = {"ticket": ticket}
                response = session.get(loinc_url, params=params, timeout=10)
        
        response.raise_for_status()
        result = response.json()
        
        # Extract information from the result
        concept_name = result.get("result", {}).get("name", "")
        
        if verbose:
            print(f"[LOINC] 获取到概念名称: {concept_name}")
        
        # LOINC name format: Component:Property:Time:System:Scale:Method
        # Property field indicates the type of measurement:
        # - MCnc = Mass Concentration (mg/dL, g/L)
        # - SCnc = Substance Concentration (mmol/L, μmol/L)
        # - NCnc = Number Concentration (10^9/L)
        # - CCnc = Catalytic Concentration (U/L for enzymes)
        # - MFr = Mass Fraction (%, mg/g)
        # - MRat = Mass Rate (mg/24h)
        
        # Parse LOINC format
        parts = concept_name.split(":")
        property_type = parts[1] if len(parts) > 1 else None
        
        # Map LOINC property types to standard SI units
        property_unit_mapping = {
            "MCnc": "g/L",           # Mass Concentration -> g/L (SI)
            "SCnc": "mmol/L",        # Substance Concentration -> mmol/L (SI)
            "NCnc": "10^9/L",        # Number Concentration (cells)
            "CCnc": "U/L",           # Catalytic Concentration (enzymes)
            "MFr": "%",              # Mass Fraction
            "MRat": "mg/24h",        # Mass Rate
            "Titr": "1",             # Titer
            "NFr": "%",              # Number Fraction
            "ACnc": "U/L",           # Arbitrary Concentration
        }
        
        unit_type = property_type
        standard_unit = property_unit_mapping.get(property_type) if property_type else None
        
        # Also check for bracket format (e.g., "[Mass/volume]")
        if not standard_unit:
            unit_match = re.search(r'\[([^\]]+)\]', concept_name)
            if unit_match:
                unit_type = unit_match.group(1)
                bracket_unit_mapping = {
                    "Mass/volume": "g/L",
                    "Moles/volume": "mmol/L",
                    "#/volume": "10^9/L",
                    "Arb'U/volume": "U/L",
                }
                standard_unit = bracket_unit_mapping.get(unit_type)
        
        # Override for specific test types based on concept name (more specific mapping)
        name_lower = concept_name.lower()
        component = parts[0].lower() if parts else name_lower
        
        # Glucose and related
        if "glucose" in component or "glu" in component:
            standard_unit = "mmol/L"
        # Lipids
        elif "cholesterol" in component or "chol" in component:
            standard_unit = "mmol/L"
        elif "triglyceride" in component:
            standard_unit = "mmol/L"
        # Liver function
        elif "bilirubin" in component:
            standard_unit = "μmol/L"
        elif "albumin" in component and "serum" in name_lower:
            standard_unit = "g/L"
        elif "protein" in component and "total" in component:
            standard_unit = "g/L"
        # Kidney function
        elif "creatinine" in component:
            standard_unit = "μmol/L"
        elif "urea" in component or "bun" in component:
            standard_unit = "mmol/L"
        elif "uric" in component or "urate" in component:
            standard_unit = "μmol/L"
        # Electrolytes
        elif "sodium" in component or "natrium" in component:
            standard_unit = "mmol/L"
        elif "potassium" in component or "kalium" in component:
            standard_unit = "mmol/L"
        elif "chloride" in component:
            standard_unit = "mmol/L"
        elif "calcium" in component:
            standard_unit = "mmol/L"
        elif "magnesium" in component:
            standard_unit = "mmol/L"
        elif "phosph" in component:  # phosphate, phosphorus
            standard_unit = "mmol/L"
        # Blood cells
        elif "hemoglobin" in component or "haemoglobin" in component:
            standard_unit = "g/L"
        elif "leukocyte" in component or "wbc" in component:
            standard_unit = "10^9/L"
        elif "platelet" in component or "thrombocyte" in component:
            standard_unit = "10^9/L"
        elif "erythrocyte" in component or "rbc" in component:
            standard_unit = "10^12/L"
        # Thyroid
        elif "thyrotropin" in component or "tsh" in component:
            standard_unit = "mIU/L"
        elif "thyroxine" in component or "t4" in component:
            standard_unit = "pmol/L"
        elif "triiodothyronine" in component or "t3" in component:
            standard_unit = "pmol/L"
        
        if standard_unit:
            if verbose:
                print(f"[LOINC] ✓ 提取单位信息: {concept_name} -> 属性类型: {unit_type}, 标准单位: {standard_unit}")
            return {
                "standard_unit": standard_unit,
                "unit_type": unit_type,
                "loinc_code": loinc_code,
                "concept_name": concept_name
            }
        
        if verbose:
            print(f"[LOINC] 无法确定标准单位: {concept_name} (属性类型: {unit_type})")
        return {"error": f"Could not determine standard unit for LOINC code: {loinc_code}"}
        
    except requests.exceptions.HTTPError as e:
        error_msg = str(e)
        if verbose:
            print(f"[LOINC] HTTP错误: {error_msg}")
        return {"error": f"HTTP error: {error_msg}"}
    except Exception as e:
        if verbose:
            print(f"[LOINC] 异常: {e}")
        return {"error": f"Failed to get LOINC info: {str(e)}"}


def normalize_unit_with_umls(
    test_name: str,
    value: float,
    unit: str,
    loinc_code: Optional[str] = None,
    verbose: bool = False
) -> Tuple[Optional[float], Optional[str], Optional[str]]:
    """
    Normalize unit using UMLS API by querying LOINC code information.
    
    This function:
    1. If LOINC code is provided, queries UMLS for standard unit
    2. If LOINC code is not provided, searches for the test name to find LOINC code first
    3. Performs unit conversion based on standard unit from LOINC
    
    Args:
        test_name: Name of the test (e.g., "血糖", "Glucose")
        value: Numerical value
        unit: Original unit (e.g., "mg/dL", "/ul")
        loinc_code: Optional LOINC code (if not provided, will search for it)
        verbose: Whether to print debug information
    
    Returns:
        Tuple of (normalized_value, normalized_unit, source)
        source indicates where the conversion came from ("umls" or None)
    """
    if not test_name or not unit:
        return value, unit, None
    
    # Common Chinese to English test name mapping for LOINC search
    test_name_mapping = {
        "血糖": "glucose",
        "空腹血糖": "fasting glucose",
        "餐后血糖": "glucose post meal",
        "糖化血红蛋白": "hemoglobin A1c",
        "总胆固醇": "total cholesterol",
        "胆固醇": "cholesterol",
        "甘油三酯": "triglycerides",
        "高密度脂蛋白": "HDL cholesterol",
        "低密度脂蛋白": "LDL cholesterol",
        "白细胞": "leukocytes",
        "红细胞": "erythrocytes",
        "血红蛋白": "hemoglobin",
        "血小板": "platelets",
        "肌酐": "creatinine",
        "尿素氮": "urea nitrogen",
        "尿酸": "uric acid",
        "谷丙转氨酶": "alanine aminotransferase",
        "谷草转氨酶": "aspartate aminotransferase",
        "总胆红素": "total bilirubin",
        "直接胆红素": "direct bilirubin",
        "白蛋白": "albumin",
        "球蛋白": "globulin",
        "钠": "sodium",
        "钾": "potassium",
        "氯": "chloride",
        "钙": "calcium",
        "镁": "magnesium",
        "磷": "phosphorus",
        "尿白蛋白": "albumin urine",
    }
    
    # Translate test name if possible
    search_name = test_name
    for cn_name, en_name in test_name_mapping.items():
        if cn_name in test_name:
            search_name = en_name
            if verbose:
                print(f"[LOINC] 检验项目名称映射: {test_name} -> {search_name}")
            break
    
    # If LOINC code is not provided, try to find it
    if not loinc_code:
        if verbose:
            print(f"[LOINC] 未提供LOINC编码，尝试搜索检验项目: {search_name}")
        
        # Search for the test name to get LOINC code
        search_result = search_umls(search_name, search_type="words")
        if "error" not in search_result:
            results = search_result.get("result", {}).get("results", [])
            if results:
                # Find result with LOINC code
                for result_item in results[:5]:  # Check top 5 results
                    cui = result_item.get("ui")
                    if cui:
                        atoms_result = get_umls_atoms(cui, source="LNC")
                        if "error" not in atoms_result:
                            atoms = atoms_result.get("result", [])
                            for atom in atoms:
                                root_source = atom.get("rootSource", "")
                                if "LNC" in root_source:
                                    code = atom.get("code", "")
                                    # Validate LOINC code format (number-checkdigit)
                                    if code and re.match(r'^\d+-\d$', code):
                                        loinc_code = code
                                        if verbose:
                                            print(f"[LOINC] 找到LOINC编码: {loinc_code} (来自atom: {atom.get('name', '')})")
                                        break
                            if loinc_code:
                                break
    
    # If we have LOINC code, query for unit information
    if loinc_code:
        unit_info = get_loinc_unit_info(loinc_code, verbose=verbose)
        if "error" not in unit_info:
            standard_unit = unit_info.get("standard_unit")
            if standard_unit:
                # Perform conversion based on standard unit
                normalized_value, normalized_unit = _convert_unit_by_standard(
                    test_name, value, unit, standard_unit, unit_info.get("unit_type")
                )
                if normalized_unit != unit:
                    if verbose:
                        print(f"[LOINC] ✓ 量纲统一成功: {value} {unit} -> {normalized_value} {normalized_unit} (LOINC标准单位: {standard_unit})")
                    return normalized_value, normalized_unit, "umls"
                elif verbose:
                    print(f"[LOINC] 原单位已是标准单位或无需转换: {unit}")
    elif verbose:
        print(f"[LOINC] 未找到LOINC编码: {test_name}")
    
    return value, unit, None


def _convert_unit_by_standard(
    test_name: str,
    value: float,
    from_unit: str,
    to_unit: str,
    unit_type: Optional[str] = None
) -> Tuple[float, str]:
    """
    Convert unit based on standard unit from UMLS/LOINC.
    
    This function performs unit conversion based on common medical unit conversions.
    
    Args:
        test_name: Name of the test
        value: Original value
        from_unit: Original unit
        to_unit: Target standard unit
        unit_type: Type of unit (e.g., "Mass/volume", "#/volume")
    
    Returns:
        Tuple of (converted_value, converted_unit)
    """
    test_name_lower = test_name.lower()
    from_unit_clean = from_unit.strip().replace(" ", "").lower()
    to_unit_clean = to_unit.strip().replace(" ", "").lower()
    
    # Normalize unit representations
    # Handle variations like mg/dL, mg/dl, mg/100ml
    from_unit_normalized = from_unit_clean.replace("mg/100ml", "mg/dl").replace("×10^", "x10^")
    to_unit_normalized = to_unit_clean.replace("μ", "u")
    
    # If units are the same (after normalization), no conversion needed
    if from_unit_normalized == to_unit_normalized:
        return value, to_unit
    
    # Glucose: mg/dL -> mmol/L (divide by 18.016)
    if ("glucose" in test_name_lower or "glu" in test_name_lower or "血糖" in test_name or
        "空腹血糖" in test_name or "餐后血糖" in test_name) and \
       "mg/dl" in from_unit_normalized and "mmol/l" in to_unit_normalized:
        return round(value / 18.016, 2), to_unit
    
    # Cholesterol (total, LDL, HDL): mg/dL -> mmol/L (divide by 38.67)
    if ("cholesterol" in test_name_lower or "chol" in test_name_lower or "胆固醇" in test_name or
        "ldl" in test_name_lower or "hdl" in test_name_lower or
        "低密度脂蛋白" in test_name or "高密度脂蛋白" in test_name) and \
       "mg/dl" in from_unit_normalized and "mmol/l" in to_unit_normalized:
        return round(value / 38.67, 2), to_unit
    
    # Triglycerides: mg/dL -> mmol/L (divide by 88.57)
    if ("triglyceride" in test_name_lower or "tg" in test_name_lower or "甘油三酯" in test_name) and \
       "mg/dl" in from_unit_normalized and "mmol/l" in to_unit_normalized:
        return round(value / 88.57, 2), to_unit
    
    # WBC: various units -> 10^9/L
    if ("wbc" in test_name_lower or "leukocyte" in test_name_lower or "白细胞" in test_name):
        # /mm3, /ul, /μL -> 10^9/L
        if ("/mm3" in from_unit_normalized or "/ul" in from_unit_normalized or "/mm³" in from_unit_clean):
            # Check if value includes multiplier (e.g., "×10^3")
            if "10^9/l" in to_unit_normalized:
                return round(value / 1000.0, 2), to_unit
        # ×10^3/mm3, ×10^3/ul -> 10^9/L
        if ("x10^3" in from_unit_normalized or "×10^3" in from_unit_clean):
            if "10^9/l" in to_unit_normalized:
                return round(value, 2), to_unit  # Same order of magnitude
    
    # Platelets: /mm3, /ul -> 10^9/L
    if ("platelet" in test_name_lower or "plt" in test_name_lower or "血小板" in test_name):
        if ("/mm3" in from_unit_normalized or "/ul" in from_unit_normalized):
            if "10^9/l" in to_unit_normalized:
                return round(value / 1000.0, 2), to_unit
        # ×10^3/ul -> 10^9/L
        if ("x10^3" in from_unit_normalized or "×10^3" in from_unit_clean):
            if "10^9/l" in to_unit_normalized:
                return round(value, 2), to_unit
    
    # RBC: /mm3, /ul -> 10^12/L
    if ("rbc" in test_name_lower or "erythrocyte" in test_name_lower or "红细胞" in test_name):
        if ("/mm3" in from_unit_normalized or "/ul" in from_unit_normalized):
            if "10^12/l" in to_unit_normalized:
                return round(value / 1000000.0, 2), to_unit
        # ×10^6/ul -> 10^12/L
        if ("x10^6" in from_unit_normalized or "×10^6" in from_unit_clean):
            if "10^12/l" in to_unit_normalized:
                return round(value, 2), to_unit
    
    # Hemoglobin: g/dL -> g/L (multiply by 10)
    if ("hemoglobin" in test_name_lower or "hb" in test_name_lower or "hgb" in test_name_lower or 
        "血红蛋白" in test_name) and \
       "g/dl" in from_unit_normalized and "g/l" in to_unit_normalized:
        return round(value * 10.0, 2), to_unit
    
    # Creatinine: mg/dL -> μmol/L (multiply by 88.4)
    if ("creatinine" in test_name_lower or "crea" in test_name_lower or "肌酐" in test_name) and \
       "mg/dl" in from_unit_normalized and ("umol/l" in to_unit_normalized or "μmol/l" in to_unit_clean):
        return round(value * 88.4, 2), to_unit
    
    # BUN/Urea: mg/dL -> mmol/L (divide by 2.8)
    if ("urea" in test_name_lower or "bun" in test_name_lower or "尿素氮" in test_name) and \
       "mg/dl" in from_unit_normalized and "mmol/l" in to_unit_normalized:
        return round(value / 2.8, 2), to_unit
    
    # Uric acid: mg/dL -> μmol/L (multiply by 59.48)
    if ("uric" in test_name_lower or "urate" in test_name_lower or "尿酸" in test_name) and \
       "mg/dl" in from_unit_normalized and ("umol/l" in to_unit_normalized or "μmol/l" in to_unit_clean):
        return round(value * 59.48, 2), to_unit
    
    # Bilirubin: mg/dL -> μmol/L (multiply by 17.1)
    if ("bilirubin" in test_name_lower or "胆红素" in test_name) and \
       "mg/dl" in from_unit_normalized and ("umol/l" in to_unit_normalized or "μmol/l" in to_unit_clean):
        return round(value * 17.1, 2), to_unit
    
    # Albumin: g/dL -> g/L (multiply by 10)
    if ("albumin" in test_name_lower or "白蛋白" in test_name) and \
       "g/dl" in from_unit_normalized and "g/l" in to_unit_normalized:
        return round(value * 10.0, 2), to_unit
    
    # Temperature: °F -> °C
    if ("temperature" in test_name_lower or "temp" in test_name_lower or "体温" in test_name) and \
       ("°f" in from_unit_normalized or "f" == from_unit_normalized) and \
       ("°c" in to_unit_normalized or "c" == to_unit_normalized):
        return round((value - 32) * 5 / 9, 2), to_unit
    
    # If no specific conversion rule matches, return original
    return value, from_unit


def clear_umls_cache() -> None:
    """Clear the UMLS ticket cache and error cache."""
    global _UMLS_TICKET, _TICKET_EXPIRY, _UMLS_AUTH_ERROR, _UMLS_AUTH_ERROR_TIME
    _UMLS_TICKET = None
    _TICKET_EXPIRY = 0
    _UMLS_AUTH_ERROR = None
    _UMLS_AUTH_ERROR_TIME = 0
