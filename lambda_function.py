import json
import os
import base64
import urllib.request
import urllib.error
import boto3
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

from datetime import datetime, timezone, timedelta


def _get_http_method(event):
    """Resolve the HTTP method from a Lambda Function URL or API Gateway event."""
    rc = event.get('requestContext') or {}
    http = rc.get('http') or {}
    if http.get('method'):
        return http['method']
    return event.get('httpMethod', 'GET')


def _load_deals_from_s3():
    """Read pipeline_deals.json from S3 and return the parsed list."""
    s3 = boto3.client('s3')
    response = s3.get_object(Bucket='pipeline-public-deal-data', Key='pipeline_deals.json')
    return json.loads(response['Body'].read().decode('utf-8'))


def _call_claude_for_matching_ids(query, deals):
    """Ask Claude which deal IDs match the user's natural-language query.

    POSTs directly to https://api.anthropic.com/v1/messages with urllib.request
    (stdlib, always available in Lambda — no external dependencies). Uses
    tool_use with a forced tool_choice for structured output, and prompt
    caching on the deals payload so repeated queries reuse the cached prefix.
    """
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

    # sort_keys makes serialization deterministic so the cached prefix is stable.
    deals_json = json.dumps(deals, sort_keys=True, default=str)

    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 4096,
        "system": [
            {
                "type": "text",
                "text": (
                    "You are a search assistant for a private secondary market deals platform. "
                    "The user will ask a question in natural language about a list of deals, and "
                    "you will be given the full deals data as JSON. Identify which deals match the "
                    "user's query and return their IDs using the return_matching_deals tool. "
                    "Match intelligently on company name, deal type (bid/offer), structure "
                    "(direct/fund/forward), price ranges, ticket sizes, management fee, carry, "
                    "stage, data room availability, highlighted status, and any other relevant "
                    "fields present in the data. If the query is broad or ambiguous, err on the "
                    "side of including possibly relevant deals. Use the exact 'id' field from each "
                    "deal as it appears in the data."
                ),
            },
            {
                "type": "text",
                "text": "DEALS DATA (JSON):\n" + deals_json,
                "cache_control": {"type": "ephemeral"},
            },
        ],
        "tools": [
            {
                "name": "return_matching_deals",
                "description": "Return the list of deal IDs that match the user's query.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "deal_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Array of matching deal IDs. Use the exact 'id' field from each deal.",
                        }
                    },
                    "required": ["deal_ids"],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "return_matching_deals"},
        "messages": [{"role": "user", "content": query}],
    }

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode('utf-8'),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            response_body = resp.read().decode('utf-8')
    except urllib.error.HTTPError as e:
        error_body = e.read().decode('utf-8', errors='replace')
        logger.error(f"Anthropic API HTTP {e.code}: {error_body}")
        raise RuntimeError(f"Anthropic API returned HTTP {e.code}: {error_body}") from e
    except urllib.error.URLError as e:
        logger.error(f"Anthropic API connection error: {e.reason}")
        raise RuntimeError(f"Anthropic API connection error: {e.reason}") from e

    try:
        result = json.loads(response_body)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON from Anthropic API: {response_body[:500]}")
        raise RuntimeError("Invalid JSON response from Anthropic API") from e

    usage = result.get('usage') or {}
    logger.info(
        "Claude search usage: input=%s cache_read=%s cache_create=%s output=%s",
        usage.get('input_tokens'),
        usage.get('cache_read_input_tokens'),
        usage.get('cache_creation_input_tokens'),
        usage.get('output_tokens'),
    )

    for block in result.get('content') or []:
        if block.get('type') == 'tool_use' and block.get('name') == 'return_matching_deals':
            raw_ids = (block.get('input') or {}).get('deal_ids') or []
            return [str(d) for d in raw_ids]

    return []


def _json_response(status_code, payload):
    return {
        'statusCode': status_code,
        'headers': {'Content-Type': 'application/json'},
        'body': json.dumps(payload),
    }


def _handle_search_post(event):
    """POST handler: run a natural-language deal search and return matching IDs."""
    body = event.get('body') or ''
    if event.get('isBase64Encoded'):
        try:
            body = base64.b64decode(body).decode('utf-8')
        except Exception as e:
            logger.error(f"Error decoding base64 body: {str(e)}")
            return _json_response(400, {'error': 'Invalid base64 body'})

    try:
        data = json.loads(body) if body else {}
    except Exception:
        return _json_response(400, {'error': 'Invalid JSON body'})

    query = (data.get('query') or '').strip()
    if not query:
        return _json_response(400, {'error': 'Missing or empty query'})

    try:
        deals = _load_deals_from_s3()
    except Exception as e:
        logger.error(f"Error reading data from S3: {str(e)}")
        return _json_response(500, {'error': f'Failed to load deals: {str(e)}'})

    try:
        matched_ids = _call_claude_for_matching_ids(query, deals)
    except Exception as e:
        logger.error(f"Error calling Claude API: {str(e)}")
        return _json_response(500, {'error': f'Search failed: {str(e)}'})

    return _json_response(200, {'deal_ids': matched_ids, 'count': len(matched_ids)})

def get_last_updated_date(deal):
    """Returns the last updated date in 'MMM D, YYYY' format or '30d+' if older than 30 days."""
    try:
        last_updated_str = deal.get('updated', '')  # Use "updated" field
        if not last_updated_str:
            return "N/A"
        
        last_updated = datetime.strptime(last_updated_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        days_since_update = (datetime.now(timezone.utc) - last_updated).days

        return "100d+" if days_since_update > 100 else last_updated.strftime('%b %-d')
    except Exception:
        return "N/A"


def format_currency(value, include_cents=False):
    try:
        float_value = float(value)
        if include_cents:
            return f"${float_value:,.2f}"
        else:
            return f"${int(float_value):,}"
    except (ValueError, TypeError):
        return value
def calculate_valuation(price, company_lr_pps, company_lr_val):
    try:
        if not price or not company_lr_pps or not company_lr_val:
            return None
        price = float(price)
        company_lr_pps = float(company_lr_pps)
        company_lr_val = float(company_lr_val)
        price_ratio = price / company_lr_pps
        new_valuation = company_lr_val * price_ratio
        
        return new_valuation
    except (ValueError, TypeError, ZeroDivisionError):
        return None

def format_valuation(valuation):
    if valuation is None:
        return ""
    return f" (${valuation:.1f}Bn)"

def lambda_handler(event, context):
    logger.info("Lambda function started")

    # Dispatch POST requests to the natural-language search handler.
    http_method = _get_http_method(event)
    if http_method == 'POST':
        return _handle_search_post(event)

    # If we're returning from Cognito with ?code=<auth_code> in the query
    # string, set the auth cookie and 302 back to the clean URL so reloads
    # don't keep the code in the address bar (and the auth modal stays
    # suppressed on subsequent visits via the cookie the client-side JS
    # already checks).
    query_params = event.get('queryStringParameters') or {}
    if query_params.get('code'):
        clean_path = (
            event.get('rawPath')
            or (event.get('requestContext') or {}).get('http', {}).get('path')
            or '/'
        )
        logger.info("Cognito auth code received; redirecting to clean URL %s", clean_path)
        return {
            'statusCode': 302,
            'headers': {
                'Location': clean_path,
                'Set-Cookie': 'CognitoIdentityServiceProvider=1; Max-Age=2592000; Path=/; Secure; SameSite=Lax',
            },
            'body': '',
        }

    # GET path: render the HTML dashboard as before.
    try:
        deals = _load_deals_from_s3()
    except Exception as e:
        logger.error(f"Error reading data from S3: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {'Content-Type': 'text/html'},
            'body': f"<h1>Error</h1><p>An error occurred while reading data: {str(e)}</p>"
        }

    # Sort by company alpha, then updated
    deals = sorted(
    deals,
    key=lambda deal: (
        deal.get('company', '').lower(),                      # A–Z
        -datetime.strptime(deal.get('updated', '1900-01-01 00:00:00'), "%Y-%m-%d %H:%M:%S").timestamp()  # Newest first
    )
    )

    # Get a unique list of companies, prioritizing highlighted ones first
    highlighted_companies = sorted(
        {deal['company'] for deal in deals if deal['company'] and deal.get('highlighted') == 'Yes'}
    )
    non_highlighted_companies = sorted(
        {deal['company'] for deal in deals if deal['company'] and deal.get('highlighted') != 'Yes'}
    )

    # Merge lists: highlighted first, then non-highlighted
    companies = highlighted_companies + non_highlighted_companies

    
    # Build the table rows
    table_rows = ""
    for deal in deals:
        stage_tooltips = {
            "Firm": "Details confirmed",
            "Inquiry": "Awaiting data",
            "Confirm": "Will confirm after bid/ask"
        }
        
        stage_html = f'<span class="stage-cell" data-tooltip="{stage_tooltips[deal["stage"]]}">{deal["stage"]}</span>'
        
        company_cell = deal['company']
        
        # Calculate valuations
        net_valuation = calculate_valuation(deal['net'], deal['company_lr_pps'], deal['company_lr_val'])
        gross_valuation = calculate_valuation(deal['gross'], deal['company_lr_pps'], deal['company_lr_val'])
        net_display = format_currency(deal['net'], include_cents=True)
        net_valuation_text = format_valuation(net_valuation).strip()  # Remove any extra whitespace
        gross_display = format_currency(deal['gross'], include_cents=True)
        gross_valuation_text = format_valuation(gross_valuation).strip()


        table_rows += f"""
        <tr class="deal-row {deal['type'].lower()} {deal['structure_class']}" data-deal-id="{deal['id']}" data-management-fee="{deal['management_fee']}" data-carry="{deal['carry']}" data-stage="{deal['stage']}" data-data-room="{deal['data_room']}" data-highlighted="{deal['highlighted']}">
            <td><a href="https://trades.graciagroup.com/deal/{deal['id']}">{deal['id']}</a></td>
            <td>{deal['type']}</td>
            <td>{company_cell}</td>
            <td>{deal['structure']}</td>
            <td class="price-cell" data-valuation="{net_valuation_text}">{net_display}</td>
            <td class="price-cell" data-valuation="{gross_valuation_text}">{gross_display}</td>
            <td>{format_currency(deal['min_deal_size'])}</td>
            <td>{format_currency(deal['max_deal_size'])}</td>
            <td>{format_currency(deal['company_lr_pps'], include_cents=True)}</td>
            <td>{format_currency(deal['company_lr_val'], include_cents=True)}</td>
            <td>{deal['management_fee']}</td>
            <td>{deal['carry']}</td>
            <td>{get_last_updated_date(deal)}</td>
        </tr>
        """

    # Create buttons for companies
    highlighted_company_buttons = " ".join([
        f"<button class='company-btn' id=\"{company}\" onclick=\"toggleCompanyFilter('{company}')\">{company}</button>"
        for company in highlighted_companies
    ])

    non_highlighted_company_buttons = " ".join([
        f"<button class='company-btn' id=\"{company}\" onclick=\"toggleCompanyFilter('{company}')\">{company}</button>"
        for company in non_highlighted_companies
    ])


    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Indications for Accredited Investors</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #f8f9fa;
                padding: 10px 20px;
                border-radius: 5px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .filter-section {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }}
            .filter-group {{
                display: flex;
                align-items: center;
                
            }}
            .filter-label {{
                font-weight: bold;
                margin-right: 5px;
            }}
            h1 {{
                color: #2c3e50;
                margin: 0 0 20px 0;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}

            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}

            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}

            thead {{
                position: sticky;
                top: 0;
                z-index: 10;
            }}

            thead tr {{
                background-color: #f8f9ff;
            }}

            thead th {{
                background-color: #f8f9ff;
                border-bottom: 3px double #ddd;
                position: sticky;
                top: 0;
                font-weight: bold;
                color: #2c3e50;
            }}

            tr:hover {{
                background-color: #f5f5f5;
            }}

            .stage-cell {{
                position: relative;
                cursor: help;
            }}

            .stage-cell:hover::after {{
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                padding: 5px 10px;
                background-color: #3498db;
                color: white;
                border-radius: 4px;
                font-size: 14px;
                white-space: nowrap;
                z-index: 20;
            }}
            
            .btn {{
                display: inline-block;
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                text-decoration: none;
                border-radius: 5px;
                transition: background-color 0.3s, box-shadow 0.3s;  /* Added shadow transition */
                margin-bottom: 10px;
                border: none;  /* Remove border */
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);  /* Add subtle shadow */
            }}

            .btn:hover {{
                background-color: #2980b9;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);  /* Slightly stronger shadow on hover */
            
            }}
            .company-buttons {{
                margin-bottom: 20px;
            }}
            .company-btn {{
                background-color: #f0f0f0;
                border: 1px solid #ddd;
                padding: 5px 10px;
                margin: 2px;
                border-radius: 3px;
                cursor: pointer;
                transition: background-color 0.3s, color 0.3s;
            }}
            .company-btn:hover {{
                background-color: #e0e0e0;
            }}
            .company-btn.active {{
                background-color: #3498db; /* Restore original blue color when clicked */
                color: white; /* Keep text readable */
            }}

            #disclaimer {{
                font-size: 0.9em;
                color: #666;
                margin-top: 30px;
            }}
            .checkbox-label {{
                display: inline-flex;
                align-items: center;
                margin-right: 15px;
                cursor: pointer;
            }}
            .checkbox-label input {{
                margin-right: 5px;
            }}

            .filter-row {{
                display: flex;
                align-items: center;
                position: relative;
                width: 100%;
            }}

            .slider-group {{
                display: flex;
                align-items: center;
                gap: 15px;
                width: 40%;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
            }}
            .right-filters {{
                display: flex;
                align-items: center;
                gap: 20px;
                margin-left: auto;
            }}

            .bottom-filter-group {{
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
                white-space: nowrap;
            }}

            .spacer {{
                width: 10px;
    
            }}
            .data-room-group {{
                position: absolute;
                left: 50%;
                transform: translateX(-50%);
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
            }}

            .highlighted-group {{
                margin-left: auto;
                border: 1px solid #ddd;
                padding: 10px;
                border-radius: 4px;
            }}

            .price-cell {{
                position: relative;
                cursor: help;
                color: #333;  /* Reset the text color to dark gray/black */
            }}

            .price-cell:hover::after {{
                content: attr(data-valuation);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                padding: 5px 10px;
                background-color: #3498db;
                color: white;  /* This is for the tooltip text */
                border-radius: 4px;
                font-size: 14px;
                white-space: nowrap;
                z-index: 20;
            }}

            .stage-cell {{
                position: relative;
                cursor: help;
                color: #333;  /* Reset the text color to dark gray/black */
            }}

            .stage-cell:hover::after {{
                content: attr(data-tooltip);
                position: absolute;
                bottom: 100%;
                left: 50%;
                transform: translateX(-50%);
                padding: 5px 10px;
                background-color: #3498db;
                color: white;  /* This is for the tooltip text */
                border-radius: 4px;
                font-size: 14px;
                white-space: nowrap;
                z-index: 20;
            }}
            .ticket-size-filter {{
                margin-bottom: 20px;
                padding: 15px;
                background-color: #f8f9fa;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}

            .slider-container {{
                display: flex;
                align-items: center;
                gap: 15px;
            }}

            #ticketSlider {{
                flex: 1;
                max-width: 300px;
            }}
            .company-filter {{
                margin-bottom: 15px; /* Add space below the company buttons */
            }}

            .toggle-btn {{
                display: inline-flex; /* Aligns with highlighted buttons */
                align-items: center; /* Centers text vertically */
                background-color: #e8f5e9; /* Light green background */
                border: 1px solid #4CAF50; /* Subtle green border */
                padding: 5px 10px;
                margin: 2px; /* Keeps spacing consistent with company buttons */
                border-radius: 3px;
                cursor: pointer;
                transition: background-color 0.3s, border 0.3s;
                white-space: nowrap; /* Prevents text wrapping */
                font-weight: bold; /* Makes text slightly stronger */
                color: #2c3e50; /* Dark text for readability */
            }}

            .toggle-btn:hover {{
                background-color: #d4edda; /* Slightly darker green on hover */
                border-color: #388E3C; /* Darker green border */
            }}


            .deal-count {{
                font-size: 28px;
                color: #2c3e50;
                font-weight: normal;
                margin-left: 5px;
                display: inline-block;
            }}

            .highlighted-group {{
                display: flex;
                align-items: center;
            }}
            .modal {{
                display: none;
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: rgba(0,0,0,0.5);
                z-index: 1000;
                opacity: 0;
                transition: opacity 0.3s ease-in-out;
            }}

            .modal-content {{
                background-color: white;
                margin: 15% auto;
                padding: 30px;
                width: 80%;
                max-width: 500px;
                border-radius: 8px;
                text-align: center;
                position: relative;
                transform: translateY(-20px);
                transition: transform 0.3s ease-in-out;
            }}

            .modal-buttons {{
                margin-top: 25px;
            }}

            .modal-btn {{
                padding: 12px 25px;
                border-radius: 5px;
                border: none;
                cursor: pointer;
                font-size: 16px;
                margin: 0 10px;
            }}

            .primary-btn {{
                background-color: #3498db;
                color: white;
            }}

            .modal.show {{
                display: block;
                opacity: 1;
            }}

            .modal.show .modal-content {{
                transform: translateY(0);
            }}

            .nl-search-container {{
                background-color: #f8f9fa;
                padding: 15px 20px;
                border-radius: 5px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .nl-search-row {{
                display: flex;
                align-items: center;
                gap: 10px;
            }}
            .nl-search-input {{
                flex: 1;
                padding: 10px 12px;
                font-size: 14px;
                border: 1px solid #ddd;
                border-radius: 4px;
                font-family: inherit;
            }}
            .nl-search-input:focus {{
                outline: none;
                border-color: #3498db;
            }}
            .nl-search-btn, .nl-clear-btn {{
                padding: 10px 18px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
                transition: background-color 0.3s;
                white-space: nowrap;
            }}
            .nl-search-btn {{
                background-color: #3498db;
                color: white;
            }}
            .nl-search-btn:hover:not(:disabled) {{
                background-color: #2980b9;
            }}
            .nl-search-btn:disabled {{
                background-color: #95a5a6;
                cursor: not-allowed;
            }}
            .nl-clear-btn {{
                background-color: #e74c3c;
                color: white;
                display: none;
            }}
            .nl-clear-btn:hover {{
                background-color: #c0392b;
            }}
            .nl-search-status {{
                margin-top: 8px;
                font-size: 13px;
                color: #666;
                min-height: 18px;
            }}

        </style>
        <!-- Core application scripts -->
        <script>
            var selectedCompanies = [];
            var searchMatchedIds = null; // Set<string> of deal IDs, or null when no search is active

            function performSearch() {{
                var input = document.getElementById('nlSearchInput');
                var query = input.value.trim();
                if (!query) return;

                var btn = document.getElementById('nlSearchBtn');
                var clearBtn = document.getElementById('nlClearBtn');
                var statusEl = document.getElementById('nlSearchStatus');

                btn.disabled = true;
                btn.textContent = 'Searching...';
                statusEl.textContent = 'Searching...';
                statusEl.style.color = '#666';

                fetch(window.location.origin + window.location.pathname, {{
                    method: 'POST',
                    headers: {{'Content-Type': 'application/json'}},
                    body: JSON.stringify({{query: query}})
                }})
                .then(function(r) {{
                    return r.json().then(function(data) {{
                        if (!r.ok) throw new Error((data && data.error) || ('HTTP ' + r.status));
                        return data;
                    }});
                }})
                .then(function(data) {{
                    if (!data || !Array.isArray(data.deal_ids)) {{
                        throw new Error('Unexpected response from search');
                    }}
                    searchMatchedIds = new Set(data.deal_ids.map(String));
                    clearBtn.style.display = 'inline-block';
                    filterTable();
                    var matchCount = searchMatchedIds.size;
                    statusEl.textContent = matchCount + ' match' + (matchCount === 1 ? '' : 'es') + ' for: "' + query + '"';
                    statusEl.style.color = '#2c3e50';
                }})
                .catch(function(err) {{
                    statusEl.textContent = 'Search failed: ' + err.message;
                    statusEl.style.color = '#c0392b';
                }})
                .finally(function() {{
                    btn.disabled = false;
                    btn.textContent = 'Search';
                }});
            }}

            function clearSearch() {{
                searchMatchedIds = null;
                document.getElementById('nlSearchInput').value = '';
                document.getElementById('nlClearBtn').style.display = 'none';
                document.getElementById('nlSearchStatus').textContent = '';
                filterTable();
            }}

            function formatDollarAmount(amount) {{
                if (amount >= 1000000) {{
                    return '$' + (amount / 1000000) + 'M';
                }} else {{
                    return '$' + (amount / 1000) + 'K';
                }}
            }}

            function toggleCompanyFilter(company) {{
                var index = selectedCompanies.indexOf(company);
                if (index > -1) {{
                    selectedCompanies.splice(index, 1);
                    document.getElementById(company).classList.remove('active');
                }} else {{
                    selectedCompanies.push(company);
                    document.getElementById(company).classList.add('active');
                }}
                filterTable();
                updateDealCount();
            }}

            function toggleNonHighlighted() {{
                var section = document.getElementById("nonHighlightedCompanies");
                var button = document.querySelector(".toggle-btn");

                if (section.style.display === "none") {{
                    section.style.display = "block";
                    button.innerHTML = "Hide All Companies ▲";
                }} else {{
                    section.style.display = "none";
                    button.innerHTML = "Show All Companies ▼";
                }}
            }}

            function updateDealCount() {{
                const visibleRows = Array.from(document.getElementsByClassName('deal-row')).filter(row => row.style.display !== 'none');
                const countElement = document.getElementById('dealCount');
                countElement.textContent = `(${{visibleRows.length}} deals)`;
            }}

            function filterTable() {{
                var buyChecked = document.getElementById('buyFilter').checked;
                var sellChecked = document.getElementById('sellFilter').checked;
                var directChecked = document.getElementById('directFilter').checked;
                var spvChecked = document.getElementById('spvFilter').checked;
                var forwardChecked = document.getElementById('forwardFilter').checked;
                var managementFeeChecked = document.getElementById('managementFeeFilter').checked;
                var carryChecked = document.getElementById('carryFilter').checked;
                var showUnconfirmedChecked = document.getElementById('showUnconfirmedFilter').checked;
                var dataRoomChecked = document.getElementById('dataRoomFilter').checked;
                var highlightedChecked = document.getElementById('highlightedFilter').checked;
                var brokerChecked = document.getElementById('brokerFilter').checked;
                
                var rows = document.getElementsByClassName('deal-row');
                
                for (var i = 0; i < rows.length; i++) {{
                    var row = rows[i];
                    var type = row.classList.contains('buy') ? 'buy' : 'sell';
                    var company = row.cells[2].innerText;
                    var isDirect = row.classList.contains('direct');
                    var isSPV = row.classList.contains('fund');
                    var isForward = row.classList.contains('forward');
                    var managementFee = parseFloat(row.getAttribute('data-management-fee')) || 0;
                    var carry = parseFloat(row.getAttribute('data-carry')) || 0;
                    var stage = row.getAttribute('data-stage');
                    var hasDataRoom = row.getAttribute('data-data-room') === 'Yes';
                    var isHighlighted = row.getAttribute('data-highlighted') === 'Yes';
                    
                    var maxDealSize = parseFloat(row.cells[7].innerText.replace(/[^0-9.-]+/g,'')) || 0;
                    var source = row.getAttribute('data-source');
                    var showBroker = !brokerChecked || (maxDealSize >= 500000 && source !== 'Notice - Co-Broker');

                    var showType = (buyChecked && type === 'buy') || (sellChecked && type === 'sell');
                    var showStructure = (directChecked && isDirect) || (spvChecked && isSPV) || (forwardChecked && isForward);
                    var showFees = (managementFeeChecked || managementFee === 0) && (carryChecked || carry === 0);
                    var showUnconfirmed = showUnconfirmedChecked || stage !== 'Confirm';
                    
                    var minDealSize = parseFloat(row.cells[6].innerText.replace(/[^0-9.-]+/g,'')) || 0;
                    var ticketSize = parseFloat(document.getElementById('ticketSlider').value);
                    var showTicketSize = minDealSize <= ticketSize;
                    
                    var show = showType && showStructure && showFees && showUnconfirmed && 
                              showTicketSize && 
                              (!dataRoomChecked || hasDataRoom) &&
                              (!highlightedChecked || isHighlighted) &&
                              showBroker;
                    
                    if (selectedCompanies.length > 0 && selectedCompanies.indexOf(company) === -1) {{
                        show = false;
                    }}

                    if (searchMatchedIds !== null) {{
                        var dealIdStr = row.getAttribute('data-deal-id');
                        if (dealIdStr === null || !searchMatchedIds.has(dealIdStr)) {{
                            show = false;
                        }}
                    }}

                    row.style.display = show ? '' : 'none';
                }}
                updateDealCount();
            }}

            document.addEventListener('DOMContentLoaded', function () {{
                    function getCookie(name) {{
                        const value = `; ${{document.cookie}}`;
                        const parts = value.split(`; ${{name}}=`);
                        if (parts.length === 2) return parts.pop().split(';').shift();
                    }}

                    const params = new URLSearchParams(window.location.search);
                    const adminKey = params.get('admin_key');
                    const isAdmin = adminKey === 'JK8h5Pq2L9aZ7rT3mN6bX' || getCookie('admin_key') === 'JK8h5Pq2L9aZ7rT3mN6bX';

                    if (adminKey === 'JK8h5Pq2L9aZ7rT3mN6bX') {{
                        document.cookie = 'admin_key=' + adminKey + '; max-age=' + (86400 * 365) + '; path=/';
                    }}

                    const cognitoCookie = getCookie('CognitoIdentityServiceProvider');
                    const hasAuthCode = params.get('code');

                    if (!isAdmin && !cognitoCookie && !hasAuthCode) {{
                        const modal = document.getElementById('authModal');
                        if (modal) {{
                            modal.classList.add('show', 'modal-force');
                            const tableElement = document.querySelector('table');
                            if (tableElement) {{
                                tableElement.style.display = 'none';
                            }}
                        }}
                    }}

                    document.getElementById('sliderValue').textContent = formatDollarAmount(document.getElementById('ticketSlider').value);
                    document.getElementById('ticketSlider').addEventListener('input', function () {{
                        const value = this.value;
                        document.getElementById('sliderValue').textContent = formatDollarAmount(parseFloat(value));
                        filterTable();
                        updateDealCount();
                    }});
                    updateDealCount();
                }});
        </script>

        <!-- PDF Generation Libraries and Script -->
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf/2.5.1/jspdf.umd.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/jspdf-autotable/3.5.31/jspdf.plugin.autotable.min.js"></script>
        <script>
            window.jsPDF = window.jspdf.jsPDF;

            function downloadPDF() {{
                const doc = new jsPDF('l', 'pt', 'a4'); // Landscape mode, points, A4 size
                const pageWidth = doc.internal.pageSize.width;
                const margin = 10; // Reduce margin to maximize width
                
                // Generate filename based on active filters
                let filename = 'GraciaGroup';
                
                // Add deal type to filename if filtered
                const buyChecked = document.getElementById('buyFilter').checked;
                const sellChecked = document.getElementById('sellFilter').checked;
                if (buyChecked && !sellChecked) {{
                    filename += '-bids';
                }} else if (sellChecked && !buyChecked) {{
                    filename += '-offers';
                }}
                
                // Add structure to filename if filtered
                const directChecked = document.getElementById('directFilter').checked;
                const spvChecked = document.getElementById('spvFilter').checked;
                const forwardChecked = document.getElementById('forwardFilter').checked;
                
                let structures = [];
                if (directChecked) structures.push('direct');
                if (spvChecked) structures.push('spv');
                if (forwardChecked) structures.push('forward');
                
                // Only add structures if not all are checked (which means no filtering)
                if (structures.length > 0 && structures.length < 3) {{
                    filename += '-' + structures.join('-');
                }}
                
                // Add ticket size to filename if not at max
                const ticketSize = parseFloat(document.getElementById('ticketSlider').value);
                if (ticketSize < 10000000) {{
                    const ticketSizeInMillions = ticketSize / 1000000;
                    filename += `-max${{ticketSizeInMillions}}M`;
                }}
                
                // Add company filter if any
                if (selectedCompanies.length > 0) {{
                    if (selectedCompanies.length <= 3) {{
                        // If 3 or fewer companies, include their names
                        filename += '-' + selectedCompanies.join('-').replace(/\s+/g, '_');
                    }} else {{
                        // If more than 3, just indicate the count
                        filename += `-${{selectedCompanies.length}}companies`;
                    }}
                }}
                
                // Add date for versioning
                const today = new Date();
                const dateStr = today.toISOString().split('T')[0]; // YYYY-MM-DD format
                filename += `-${{dateStr}}`;
                
                // Clean up filename - replace spaces and special characters
                filename = filename.replace(/[^\w-]/g, '_').toLowerCase();

                function addHeaderAndFooter() {{
                    doc.setFontSize(10);
                    doc.setTextColor(52, 152, 219);
                    doc.text('Prepared by Chad Gracia • cgracia@rainmakersecurities.com • +1-917-549-8969', margin, 20);
                    
                    doc.setFontSize(8);
                    doc.setTextColor(127, 140, 141);
                    doc.text('To see full interactive report, visit: https://trades.graciagroup.com/', margin, doc.internal.pageSize.height - 20);
                }}

                try {{
                    doc.setFont("helvetica");
                    doc.setFontSize(22);
                    doc.setTextColor(44, 62, 80);
                    doc.text('Private Secondary Indications', margin, 50);
                    
                    doc.setFontSize(12);
                    doc.setTextColor(127, 140, 141);
                    doc.text(`Generated: ${{new Date().toLocaleString()}}`, margin, 70);

                    const visibleRows = Array.from(document.querySelectorAll('.deal-row'))
                        .filter(row => row.style.display !== 'none')
                        .map(row => Array.from(row.cells).map(cell => cell.innerText));

                    const columns = [
                        'Deal ID', 'Type', 'Company', 'Structure', 'Net', 'Gross', 
                        'Min Deal Size', 'Max Deal Size', 'Company LR (PPS)', 
                        'Company LR Val ($Bn)', 'Man. Fee', 'Carry', 'Updated'
                    ];
                    
                    doc.autoTable({{
                        startY: 100,
                        head: [columns],
                        body: visibleRows,
                        styles: {{
                            fontSize: 9,
                            cellPadding: 4,
                        }},
                        headStyles: {{
                            fillColor: [44, 62, 80],
                            fontSize: 10,
                            halign: 'center',
                            textColor: [255, 255, 255] // White text on dark header
                        }},
                        margin: {{ left: 20, right: 20 }}, // Keeps table within margins
                        tableWidth: "auto", // Automatically adjusts column widths
                        didDrawPage: function(data) {{
                            addHeaderAndFooter();
                        }}
                    }});

                    doc.addPage();
                    doc.setFontSize(14);
                    doc.setTextColor(44, 62, 80);
                    doc.text('DISCLAIMER', margin, 50);
                    
                    doc.setFontSize(10);
                    doc.setTextColor(127, 140, 141);
                    const disclaimer = document.getElementById('disclaimer').innerText;
                    const splitDisclaimer = doc.splitTextToSize(disclaimer, pageWidth - (margin * 2));
                    doc.text(splitDisclaimer, margin, 70);

                    addHeaderAndFooter();
                    
                    doc.save(`${{filename}}.pdf`);
                    
                }} catch (error) {{
                    console.error("Error generating PDF:", error);
                }}
            }}

        </script>
    </head>
    <body>
        <button class="btn" onclick="window.location.href='https://www.graciagroup.com/'">Gracia Group Home</button>
        <button class="btn" onclick="window.location.href='https://6dzzw7nvdqtulz3hrtux3ofr440jbjho.lambda-url.us-east-1.on.aws/'">Create Watchlist</button>
        <button class="btn" onclick="downloadPDF()">Download PDF</button>
        <button class="btn" onclick="location.reload()">Refresh</button>

        <div class="nl-search-container">
            <div class="nl-search-row">
                <input type="text" id="nlSearchInput" class="nl-search-input" placeholder="Ask a question about the deals (e.g., 'SpaceX offers under $50M ticket size')" onkeydown="if(event.key==='Enter'){{performSearch()}}">
                <button id="nlSearchBtn" class="nl-search-btn" onclick="performSearch()">Search</button>
                <button id="nlClearBtn" class="nl-clear-btn" onclick="clearSearch()">Clear search</button>
            </div>
            <div id="nlSearchStatus" class="nl-search-status"></div>
        </div>

        <div class="header">
            <h1>Indications for Accredited Investors <span id="dealCount" class="deal-count"></span></h1>
            <div class="filter-section">
                <div class="filter-group">
                    <span class="filter-label">Type:</span>
                    <label class="checkbox-label"><input type="checkbox" id="buyFilter" checked onchange="filterTable()">Show Bids</label>
                    <label class="checkbox-label"><input type="checkbox" id="sellFilter" checked onchange="filterTable()">Show Offers</label>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Structure:</span>
                    <label class="checkbox-label"><input type="checkbox" id="directFilter" checked onchange="filterTable()"> Direct</label>
                    <label class="checkbox-label"><input type="checkbox" id="spvFilter" checked onchange="filterTable()"> Fund</label>
                    <label class="checkbox-label"><input type="checkbox" id="forwardFilter" checked onchange="filterTable()"> Forward</label>
                </div>
                <div class="filter-group">
                    <span class="filter-label">Fees:</span>
                    <label class="checkbox-label"><input type="checkbox" id="managementFeeFilter" checked onchange="filterTable()"> Management</label>
                    <label class="checkbox-label"><input type="checkbox" id="carryFilter" checked onchange="filterTable()"> Carry</label>
                </div>
                <div class="filter-group">
                    <label class="checkbox-label"><input type="checkbox" id="showUnconfirmedFilter" checked onchange="filterTable()"> Show Unconfirmed Orders</label>
                </div>

            </div>
        </div>
        <div class="ticket-size-filter">
            <div class="filter-row">
                <div class="slider-group">
                    <strong>My ticket size:</strong>
                    <input type="range" id="ticketSlider" min="100000" max="10000000" step="100000" value="10000000">
                    <span id="sliderValue">$10M</span>
                </div>
                <div class="right-filters">
                    <div class="bottom-filter-group">
                        <label class="checkbox-label"><input type="checkbox" id="dataRoomFilter" onchange="filterTable()"> Data Room</label>
                    </div>
                    <div class="spacer"></div>
                    <div class="bottom-filter-group">
                        <label class="checkbox-label"><input type="checkbox" id="brokerFilter" onchange="filterTable()"> Available for Brokers</label>
                    </div>
                    <div class="spacer"></div>
                    <div class="bottom-filter-group">
                        <label class="checkbox-label"><input type="checkbox" id="highlightedFilter" onchange="filterTable()"> Highlighted Deals</label>
                    </div>
                </div>
            </div>
            </div>
        </div>
        <div class="company-filter">
            <strong>Highlighted Companies:</strong>

            <!-- Always visible: Highlighted Companies -->
            <div id="highlightedCompanies">
                {highlighted_company_buttons}
                <button class="toggle-btn" onclick="toggleNonHighlighted()">Show All Companies ▼</button>
            </div>

            <div id="nonHighlightedCompanies" style="display: none;">
                {non_highlighted_company_buttons}
            </div>

        </div>

        <table id="dealsTable">
            <thead>
                <tr>
                    <th>Deal ID</th>
                    <th>Type</th>
                    <th>Company</th>
                    <th>Structure</th>
                    <th>Net</th>
                    <th>Gross</th>
                    <th>Min Deal Size</th>
                    <th>Max Deal Size</th>
                    <th>Company LR (PPS)</th>
                    <th>Company LR Val ($Bn)</th>
                    <th>Man. Fee</th>
                    <th>Carry</th>
                    <th>Updated</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <div id="disclaimer">
            <p>DISCLOSURE: Chad Gracia ("Gracia") is a principal of The Gracia Group, LLC ("Gracia Group") and a registered agent of Rainmaker Securities, LLC ("RMS"). Gracia Group is a consulting firm and outside business activity of Gracia. Gracia Group is not affiliated with RMS. Rainmaker Securities, LLC ("RMS") is a FINRA registered broker-dealer and SIPC member. Find this broker-dealer and its agents on BrokerCheck. Our relationship summary can be found on the RMS website.</p>
            <p>RMS is engaged by its clients to make referrals to buyers or sellers of private securities ("Securities"). If such client closes a Securities transaction with a buyer or seller so referred, RMS is entitled to a success fee from the client. Such success fee may be in the form of cash or in warrants to purchase securities of the client or client's affiliate. RMS or RMS representatives may hold equity in its issuer clients or in the issuers of securities purchased or sold by the parties to a transaction.</p>
            <p>This communication is confidential and is addressed only to its intended recipient. This communication does not represent an offer or solicitation to buy or sell Securities. Such an offer must be made via definitive legal documentation by the seller of securities.</p>
            <p>Investments in the Securities are speculative and involve a high degree of risk. An investor in the Securities should have little to no need for liquidity in the foreseeable future and have sufficient finances to withstand the loss of the entire investment.</p>
            <p>RMS does not recommend the purchase or sale of Securities. Potential buyers or sellers of the Securities should seek professional counsel prior to entering into any transaction.</p>
            <p>Chad Gracia is a registered agent of Rainmaker Securities, LLC (“RMS”) and a principal of Gracia Group. RMS is a FINRA registered broker-dealer and SIPC member. Find RMS and its agents on BrokerCheck. The RMS relationship summary can be found on the RMS website.  RMS is not an affiliate of Gracia Group. All securities transactions conducted by Chad Gracia will be conducted via RMS.</p>
        </div>
        <div id="authModal" class="modal">
            <div class="modal-content">
                <h2>Private Secondary Indications</h2>
                <p>This platform provides research and pricing for accredited investors.</p>
                <div class="modal-buttons">
                    <a href="https://us-east-1dsttcaqx7.auth.us-east-1.amazoncognito.com/login?client_id=71vrglkidm13jb73u7nje3d1t2&response_type=code&redirect_uri=https://trades.graciagroup.com" 
                        class="modal-btn primary-btn">Sign In or Register</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return {
        'statusCode': 200,
        'headers': {'Content-Type': 'text/html'},
        'body': html_content
    }
