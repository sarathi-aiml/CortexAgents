# Imports
import streamlit as st
from snowflake.snowpark.session import Session
import requests # For making HTTP requests to the Snowflake API
import os
import json
import copy # To deep copy message list for filtering
import re # For citation and related query handling
import pandas as pd # To handle SQL results
from datetime import datetime # For debug log timestamps
from typing import Generator #, Tuple, List, Any # Import Generator and other types

# --- Snowflake Connection Configuration ---
# !!! IMPORTANT: Replace placeholders below with your actual Snowflake details !!!
SNOWFLAKE_ACCOUNT = "xy12345.us-west-2" # e.g., xy12345.us-west-2
SNOWFLAKE_USER = "USER_Name"
SNOWFLAKE_PASSWORD="PAT"
# --- Key Pair Auth Details ---

# --- Other Snowflake Config (Optional but Recommended) ---
SNOWFLAKE_WAREHOUSE = "INSURANCEWAREHOUSE"
SNOWFLAKE_DATABASE = "INSURANCEDB"
SNOWFLAKE_SCHEMA = "DATA"
SNOWFLAKE_ROLE = "ACCOUNTADMIN" # Optional: Specify role if needed

# --- API Configuration ---
# Construct the base URL for Snowflake API calls
# Update SNOWFLAKE_ACCOUNT first for this to be correct
SNOWFLAKE_API_BASE_URL = f"https://{SNOWFLAKE_ACCOUNT}.snowflakecomputing.com" if SNOWFLAKE_ACCOUNT != "YOUR_SNOWFLAKE_ACCOUNT_IDENTIFIER" else "https://<YOUR_ACCOUNT>.snowflakecomputing.com"

API_TIMEOUT_SECONDS = 600 # Timeout for requests library (in seconds)
MODEL_NAME = 'claude-3-5-sonnet'

# Citation/Reltaed Queries regex
RELATED_QUERIES_REGEX = r"Related query:\s*(.*?)\s*Answer:\s*(.*?)(?=\nRelated query:|\n*$)"
CITATION_REGEX = r"【†(\d+)†】"

# Define Tools
CORTEX_ANALYST_TOOL_DEF = { "tool_spec": { "type": "cortex_analyst_text_to_sql", "name": "analyst1" } }
CORTEX_SEARCH_TOOL_DEF = { "tool_spec": { "type": "cortex_search", "name": "search1" } }
SQL_EXEC_TOOL_DEF = { "tool_spec": { "type": "sql_exec", "name": "sql_exec" } }
DATA_TO_CHART_TOOL_DEF = { "tool_spec": { "type": "data_to_chart", "name": "data_to_chart" } }

# Define the list of tools to be sent to the API.
AGENT_TOOLS = [
    CORTEX_ANALYST_TOOL_DEF, 
    CORTEX_SEARCH_TOOL_DEF,
    SQL_EXEC_TOOL_DEF, 
    DATA_TO_CHART_TOOL_DEF,
]

# Define Tool Resources, ensure paths/names are valid in your Snowflake account)
# !!! IMPORTANT: Replace placeholder values below !!!
AGENT_TOOL_RESOURCES = {
    "analyst1": { "semantic_model_file": "@INSURANCEDB.DATA.CLAIM_STORAGE/customer_semantic_model.yaml" },
    "search1": { "name": "support_docs_search", "max_results": 10 },
}

# Define Experimental Params
AGENT_EXPERIMENTAL_PARAMS = {
    "EnableRelatedQueries": True 
}


# Set page title and icon
st.set_page_config(page_title="Cortex Agent Chat (Standalone)", page_icon="❄️", layout="wide")

# --- Initialize Session State ---
if "messages" not in st.session_state: st.session_state.messages = []
if "debug_log" not in st.session_state: st.session_state.debug_log = []
if "debug_mode" not in st.session_state: st.session_state.debug_mode = False
if "snowpark_session" not in st.session_state: st.session_state.snowpark_session = None # Store session

# --- Helper: Add Log Entry (Conditional) ---
# (No changes needed in this function itself)
def add_log(level, message):
    """Appends a formatted message to the debug log if debug mode is ON."""
    # Check toggle state directly
    if not st.session_state.get("debug_mode", False): return
    # Initialize log if it doesn't exist (shouldn't be needed with init above, but safe)
    if "debug_log" not in st.session_state: st.session_state.debug_log = []
    # Ensure message is string
    if not isinstance(message, str):
        try: message = json.dumps(message, indent=2)
        except TypeError: message = str(message) # Fallback to string conversion
    log_entry = f"{datetime.now()} - {level}: {message}"
    st.session_state.debug_log.append(log_entry)

def create_snowpark_session():
    """Creates and returns a Snowpark session using username/password authentication."""
    add_log("INFO", "Attempting to create Snowpark session using username/password...")

    if not SNOWFLAKE_ACCOUNT or not SNOWFLAKE_USER or not SNOWFLAKE_PASSWORD:
        add_log("ERROR", "Missing Snowflake ACCOUNT, USER, or PASSWORD.")
        st.sidebar.error("Fatal Error: Snowflake ACCOUNT, USER, or PASSWORD is not set.")
        return None

    connection_parameters = {
        "account": SNOWFLAKE_ACCOUNT,
        "user": SNOWFLAKE_USER,
        "password": SNOWFLAKE_PASSWORD,
        **({"warehouse": SNOWFLAKE_WAREHOUSE} if SNOWFLAKE_WAREHOUSE else {}),
        **({"database": SNOWFLAKE_DATABASE} if SNOWFLAKE_DATABASE else {}),
        **({"schema": SNOWFLAKE_SCHEMA} if SNOWFLAKE_SCHEMA else {}),
        **({"role": SNOWFLAKE_ROLE} if SNOWFLAKE_ROLE else {}),
    }

    
    loggable_params = {k: v for k, v in connection_parameters.items() if k != 'password'}
    add_log("DEBUG", f"Snowpark Connection Parameters: {loggable_params}")

    try:
        
        session = Session.builder.configs(connection_parameters).create()
        
        add_log("SUCCESS", "Snowpark session created successfully.")
        st.sidebar.success("Snowpark session active!")
        return session
    except Exception as e:
        add_log("ERROR", f"Failed to create Snowpark session: {e}")
        st.sidebar.error(f"Could not connect to Snowflake. Please check credentials. Error: {e}")
        return None



# --- Get or Create Session ---
if st.session_state.snowpark_session is None:
    if SNOWFLAKE_ACCOUNT != "YOUR_SNOWFLAKE_ACCOUNT_IDENTIFIER" and SNOWFLAKE_USER != "YOUR_SNOWFLAKE_USER" and PRIVATE_KEY_PATH != "/path/to/your/rsa_key.p8":
        st.session_state.snowpark_session = create_snowpark_session()
    else:
        st.warning("Snowflake connection details placeholders are not filled in. Cannot establish session.")
        add_log("WARN", "Skipping session creation due to placeholder values.")

session = st.session_state.snowpark_session if st.session_state.snowpark_session else None

# --- Sidebar Setup ---
st.sidebar.title("Controls")
# Use default key generation for toggle, relying on label
st.session_state.debug_mode = st.sidebar.toggle("Enable Debug Mode", value=st.session_state.get("debug_mode", False))
# Add explicit display of the state for debugging the toggle itself
st.sidebar.write(f"(Debug Mode State: {st.session_state.debug_mode})")

if st.sidebar.button("Clear Chat History & Log"):
    st.session_state.messages = []
    st.session_state.debug_log = []
    st.rerun()

# --- Application Title and Header ---
st.title("❄️ Chat with Cortex Agent")

# --- Helper Functions ---
def filter_messages_for_api(messages):
    """
    Creates a deep copy of messages, including assistant messages,
    filtering out only display-specific elements like fetched_table.
    """
    messages_copy = []
    types_to_remove = ["fetched_table"] # Only remove table markdown added client-side

    for msg in messages:
        msg_copy = copy.deepcopy(msg)
        if msg_copy["role"] == "assistant":
            # Filter specific content part types from assistant messages if needed
            if isinstance(msg_copy.get("content"), list):
                 msg_copy["content"] = [
                     part for part in msg_copy["content"]
                     if part.get("type", "").lower() not in types_to_remove
                 ]
        messages_copy.append(msg_copy)

    add_log("TRACE", f"Messages prepared for API: {messages_copy}")
    return messages_copy
# Replace your stream_text_and_collect_parts function with this improved version:

def stream_text_and_collect_parts(sse_response, full_text: list, non_text_collector: list) -> Generator[str, None, str]:
    """
    Directly process the streaming Response without using SSEClient.
    Works with raw Response objects from requests library.
    """
    # Clear the lists at the start
    full_text.clear()
    non_text_collector.clear()
    full_text_accumulator = ""
    
    try:
        # Check if we have a proper Response object
        if hasattr(sse_response, 'iter_lines') and callable(getattr(sse_response, 'iter_lines')):
            add_log("DEBUG", "Processing Response.iter_lines() directly")
            
            # Variables to track current event processing
            current_event = None
            event_data = ""
            
            # Process the response line by line
            for line in sse_response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                
                # Ensure we have a string
                line_str = line if isinstance(line, str) else line.decode('utf-8')
                
                add_log("TRACE", f"Raw line: {line_str[:50]}...")
                
                # Parse SSE format
                if line_str.startswith('event:'):
                    # Complete previous event if we have one
                    if current_event == 'message.delta' and event_data:
                        # Process the completed event data
                        text_chunks = extract_text_from_message_delta(event_data, non_text_collector)
                        for text in text_chunks:
                            full_text_accumulator += text
                            yield text
                    
                    # Start a new event
                    current_event = line_str[6:].strip()
                    event_data = ""
                    add_log("DEBUG", f"New event: {current_event}")
                
                elif line_str.startswith('data:'):
                    # Append to current event data
                    data_content = line_str[5:].strip()
                    event_data = data_content
                    
                    # Handle 'done' event immediately
                    if current_event == 'done' and data_content == '[DONE]':
                        add_log("DEBUG", "Received 'done' event, stopping")
                        break
                
                # Process message.delta events as they come in
                if current_event == 'message.delta' and event_data and event_data != '[DONE]':
                    # Process this event
                    text_chunks = extract_text_from_message_delta(event_data, non_text_collector)
                    for text in text_chunks:
                        full_text_accumulator += text
                        yield text
                    
                    # Reset for next event
                    event_data = ""
        else:
            # Not a proper Response object
            error_msg = f"Invalid response type: {type(sse_response)}"
            add_log("ERROR", error_msg)
            yield f"Error: {error_msg}"
    
    except Exception as e:
        error_msg = f"Error processing stream: {e}"
        add_log("ERROR", error_msg)
        yield f"\n\n{error_msg}"
    
    finally:
        # Ensure the response is closed
        if hasattr(sse_response, 'close') and callable(getattr(sse_response, 'close')):
            sse_response.close()
    
    # Store the complete text
    full_text.append(full_text_accumulator)
    
    return full_text_accumulator

def extract_text_from_message_delta(event_data, non_text_collector):
    """Extract text chunks from a message.delta event data."""
    text_chunks = []
    
    try:
        # Parse JSON data
        delta_data = json.loads(event_data)
        add_log("TRACE", f"Parsed delta data: {str(delta_data)[:100]}...")
        
        # Extract content items
        content_items = delta_data.get('delta', {}).get('content', [])
        
        for item in content_items:
            item_type = item.get('type')
            
            if item_type == 'text':
                # Direct text content
                text = item.get('text', '')
                if text:
                    add_log("DEBUG", f"Found text content: {text[:50]}...")
                    text_chunks.append(text)
            
            elif item_type == 'tool_results':
                # Store the tool result
                non_text_collector.append(item)
                add_log("DEBUG", f"Collected tool_results: {item.get('tool_results', {}).get('name', 'unknown')}")
                
                # Check for text in analyst tool results
                tool_results = item.get('tool_results', {})
                tool_name = tool_results.get('name')
                
                if tool_name == 'analyst1':
                    try:
                        content_list = tool_results.get('content', [])
                        if content_list and isinstance(content_list, list):
                            first_content = content_list[0]
                            
                            # Extract JSON
                            json_data = None
                            if isinstance(first_content, dict):
                                json_data = first_content.get('json', {})
                            elif isinstance(first_content, str):
                                try:
                                    parsed = json.loads(first_content)
                                    if isinstance(parsed, dict):
                                        json_data = parsed.get('json', {})
                                except:
                                    pass
                            
                            # Check for text field
                            if isinstance(json_data, dict) and 'text' in json_data:
                                analyst_text = json_data.get('text', '')
                                if analyst_text:
                                    add_log("DEBUG", f"Found analyst text: {analyst_text[:50]}...")
                                    text_chunks.append(analyst_text)
                    except Exception as e:
                        add_log("WARN", f"Error extracting text from analyst tool: {e}")
            
            elif item_type in ['tool_use', 'chart', 'table', 'fetched_table']:
                # Collect other non-text parts
                item_copy = {k: v for k, v in item.items() if k != 'index'}
                if item_type == 'fetched_table':
                    item_copy.setdefault('toolResult', False)
                non_text_collector.append(item_copy)
                add_log("DEBUG", f"Collected {item_type}")
    
    except json.JSONDecodeError as e:
        add_log("ERROR", f"JSON parse error: {e} - Data: {event_data[:100]}")
    except Exception as e:
        add_log("ERROR", f"Error processing message delta: {e}")
    
    return text_chunks

def process_message_delta(event_data, non_text_collector):
    """
    Process a message.delta event data and extract all text chunks.
    Returns a list of text chunks to yield.
    Collects non-text parts in the provided list.
    """
    text_chunks = []
    
    try:
        # Parse JSON data
        try:
            event_data_obj = json.loads(event_data)
        except json.JSONDecodeError as e:
            add_log("ERROR", f"Failed to parse JSON: {e}, Data: {event_data[:100]}...")
            return []
            
        # Get content list from delta
        delta_content = event_data_obj.get("delta", {}).get("content", [])
        
        # Process each content part
        for part in delta_content:
            part_type = part.get("type")
            
            if part_type == "text":
                # Extract and yield text
                text = part.get("text", "")
                if text:
                    # Only filter related queries if the text is substantial
                    if len(text) > 20 and re.search(RELATED_QUERIES_REGEX, text, re.DOTALL | re.IGNORECASE):
                        add_log("TRACE", f"Filtered related query text: {text[:50]}...")
                    else:
                        text_chunks.append(text)
            
            elif part_type == "tool_results":
                # Collect tool results
                non_text_collector.append(part)
                
                # Try to extract text from analyst results
                try:
                    tool_results = part.get("tool_results", {})
                    if tool_results.get("name") == "analyst1":
                        contents = tool_results.get("content", [])
                        if contents and isinstance(contents, list):
                            content_item = contents[0]
                            json_data = None
                            
                            if isinstance(content_item, dict):
                                json_data = content_item.get("json", {})
                            elif isinstance(content_item, str):
                                try:
                                    json_data = json.loads(content_item)
                                    if isinstance(json_data, dict):
                                        json_data = json_data.get("json", {})
                                except:
                                    pass
                            
                            if isinstance(json_data, dict) and "text" in json_data:
                                analyst_text = json_data.get("text", "")
                                if analyst_text and not re.search(RELATED_QUERIES_REGEX, analyst_text, re.DOTALL | re.IGNORECASE):
                                    text_chunks.append(analyst_text)
                                    add_log("DEBUG", f"Extracted analyst text: {analyst_text[:50]}...")
                except Exception as e:
                    add_log("WARN", f"Error extracting text from analyst tool: {e}")
            
            elif part_type in ["tool_use", "chart", "table", "fetched_table"]:
                # Collect other non-text content
                content_part_data = {k: v for k, v in part.items() if k != 'index'}
                if part_type == "fetched_table":
                    content_part_data.setdefault("toolResult", False)
                non_text_collector.append(content_part_data)
                add_log("DEBUG", f"Collected non-text part: {part_type}")
    
    except Exception as e:
        add_log("ERROR", f"Error processing message delta: {e}")
    
    return text_chunks

def process_sse_event(event_type, event_data, current_text_accumulator, non_text_collector):
    """
    Process a single SSE event and extract text/non-text content.
    Returns tuple of (text_delta, text_to_yield, new_accumulator) or None if no text to yield.
    """
    if event_type != 'message.delta':
        return None
    
    try:
        if not event_data or event_data == '[DONE]':
            return None
            
        # Parse the JSON data
        event_data_obj = json.loads(event_data)
        delta_content_list = event_data_obj.get("delta", {}).get("content", [])
        
        for part in delta_content_list:
            part_type = part.get("type")
            if not part_type:
                continue
                
            # Variables to track text deltas
            text_delta_to_yield = None
            text_delta_for_accumulator = None
            
            # Handle text content
            if part_type == "text":
                text_delta = part.get("text", "")
                if text_delta:
                    text_delta_for_accumulator = text_delta
                    # Filter out related queries
                    if not re.search(RELATED_QUERIES_REGEX, text_delta, re.DOTALL | re.IGNORECASE):
                        text_delta_to_yield = text_delta
                        
            # Handle tool results - need to extract text from certain tools
            elif part_type == "tool_results":
                tool_results = part.get("tool_results", {})
                tool_name = tool_results.get("name")
                
                # Extract text from analyst1 tool results if present
                if tool_name == "analyst1":
                    try:
                        content_items = tool_results.get('content', [])
                        if content_items and isinstance(content_items, list):
                            content_item = content_items[0]
                            json_data = None
                            
                            # Extract JSON data depending on format
                            if isinstance(content_item, dict):
                                json_data = content_item.get('json', {})
                            elif isinstance(content_item, str):
                                try:
                                    json_data = json.loads(content_item).get('json', {})
                                except json.JSONDecodeError:
                                    pass
                                    
                            # Extract text from JSON if present
                            if isinstance(json_data, dict) and "text" in json_data:
                                analyst_text = json_data.get("text", "")
                                if analyst_text:
                                    text_delta_for_accumulator = analyst_text
                                    if not re.search(RELATED_QUERIES_REGEX, analyst_text, re.DOTALL | re.IGNORECASE):
                                        text_delta_to_yield = analyst_text
                    except Exception as e:
                        add_log("WARN", f"Error extracting text from analyst1 tool: {e}")
                
                # Always collect tool results
                non_text_collector.append(part)
                
            # Collect other non-text content
            elif part_type in ["tool_use", "chart", "table", "fetched_table"]:
                # Clone the part excluding index
                content_part_data = {k: v for k, v in part.items() if k != 'index'}
                # Add toolResult flag for fetched_table
                if part_type == "fetched_table":
                    content_part_data.setdefault("toolResult", False)
                non_text_collector.append(content_part_data)
            else:
                add_log("WARN", f"Unhandled content type: {part_type}")
                
            # Return any text we found to be yielded
            if text_delta_for_accumulator:
                new_accumulator = current_text_accumulator + text_delta_for_accumulator
                return (text_delta_for_accumulator, text_delta_to_yield, new_accumulator)
                
    except json.JSONDecodeError as e:
        add_log("ERROR", f"JSON decode error in event data: {e}")
    except Exception as e:
        add_log("ERROR", f"Error processing event: {e}")
        
    return None

def execute_sql(sql_query: str) -> tuple[str | None, pd.DataFrame | None]:
    """ Executes SQL (after cleaning), gets query ID via query_history and results DataFrame. """
    global session
    if not session:
        add_log("ERROR", "Snowpark session not available for execute_sql.")
        return None, None

    query_id, dataframe = None, None
    add_log("DEBUG", f"Original SQL received: ```sql\n{sql_query}\n```")
    cleaned_sql = sql_query.strip()
    if cleaned_sql.endswith(';'): 
        cleaned_sql = cleaned_sql[:-1].rstrip()
    comment_to_remove = "-- Generated by Cortex Analyst"
    if comment_to_remove in cleaned_sql: 
        cleaned_sql = cleaned_sql.replace(comment_to_remove, "").strip()
    
    add_log("INFO", f"Attempting to execute cleaned SQL...")
    add_log("DEBUG", f"Cleaned SQL:\n```sql\n{cleaned_sql}\n```")

    try:
        add_log("DEBUG", "Executing SQL within session.query_history context...")
        with session.query_history(True) as query_history:
            snowpark_df = session.sql(cleaned_sql)
            dataframe = snowpark_df.to_pandas()
        add_log("DEBUG", "Finished SQL execution within context.")

        if query_history.queries:
            query_id = query_history.queries[-1].query_id
            add_log("SUCCESS", f"SQL OK. Query ID from history: {query_id}")
        else:
            add_log("WARN", "Query executed but query_history was empty.")
            query_id = session.last_query_id
            add_log("WARN", f"Using fallback session.last_query_id: {query_id}")

        if dataframe is None or dataframe.empty: add_log("INFO", "Query returned no results.")
        return query_id, dataframe

    except Exception as e:
        error_message = f"Error executing SQL: {e}\nQuery attempted:\n```sql\n{cleaned_sql}\n```"
        add_log("ERROR", error_message)
        last_qid = session.last_query_id if session else None
        add_log("INFO", f"Attempting to get last query ID after error: {last_qid}")
        return last_qid, None

def get_sql_exec_user_message(query_id: str) -> dict:
    """ Constructs the user message for SQL results (without ID). """
    return {
        "role": "user",
        "content": [{"type": "tool_results", "tool_results": {
            "name": "sql_exec",
            "content": [{"type": "json", "json": { "query_id": query_id }}]
        }}],
    }

def call_agent_api(messages_to_send: list, call_label: str) -> requests.Response | None:
    """
    Calls agent API using requests and the Snowpark session token (as requested).
    Includes "stream": True in the request body.
    Returns the streaming requests.Response object on success (status 200),
    otherwise returns None after logging the error.
    """
    global session 
    api_path = "/api/v2/cortex/agent:run"
    api_url = f"{SNOWFLAKE_API_BASE_URL}{api_path}"
    add_log("INFO", f"Making {call_label} API call to: {api_url}")

    session_token = None
    if not session:
        add_log("ERROR", f"{call_label} API call failed: Snowpark session is not available.")
        return None
    try:
        session_token = session.conf.get("rest").token
        if not session_token:
             raise ValueError("Retrieved session token is empty.")
        add_log("DEBUG", f"{call_label} Retrieved session token successfully (token value not logged).")
    except Exception as e:
        add_log("ERROR", f"{call_label} API call failed: Could not retrieve token from Snowpark session: {e}")
        return None

    # Use the REVERTED filter_messages_for_api
    messages_for_api = filter_messages_for_api(messages_to_send)
    request_body = {
        "model": MODEL_NAME,
        "messages": messages_for_api, # Now includes assistant context
        "tools": AGENT_TOOLS,
        "tool_resources": AGENT_TOOL_RESOURCES,
        "experimental": AGENT_EXPERIMENTAL_PARAMS,
        "stream": True # Need this here so Analyst responds with a stream...
    }
    headers = {
        "Authorization": f'Snowflake Token="{session_token}"',
        "Content-Type": "application/json",
        "Accept": "*/*", # Accept any response type (including text/event-stream)
    }

    add_log("DEBUG", f"{call_label} Request Body: {json.dumps(request_body, indent=2)}")
    loggable_headers = {k: (v[:18] + '..."' if k == "Authorization" else v) for k, v in headers.items()}
    add_log("DEBUG", f"{call_label} Request Headers: {loggable_headers}")

    try:
        # Make the request with stream=True
        response = requests.post(
            api_url,
            headers=headers,
            json=request_body,
            timeout=API_TIMEOUT_SECONDS,
            stream=True # IMPORTANT: Keep the connection open for streaming
        )
        add_log("DEBUG", f"{call_label} Raw API Response Status: {response.status_code}")

        # Check status code BEFORE returning the response object
        if response.status_code == 200:
            add_log("INFO", f"{call_label} API call successful (200). Returning response object for streaming.")
            return response # Return the response object for the caller to handle streaming
        else:
            # Read the error body if not 200
            try:
                 error_text = response.text # Read the full error text
            except Exception as read_err:
                 error_text = f"(Could not read error response body: {read_err})"
            error_details = f"Status Code: {response.status_code}. Response: {error_text}"
            add_log("ERROR", f"{call_label} API request failed: {error_details}")
            response.close() # Close the connection if not returning the object
            return None # Indicate failure

    except requests.exceptions.Timeout:
        add_log("ERROR", f"Error during {call_label} API call: Request timed out after {API_TIMEOUT_SECONDS} seconds.")
        return None
    except requests.exceptions.RequestException as e:
        add_log("ERROR", f"Error during {call_label} API call (requests exception): {e}")
        return None
    except Exception as e:
        add_log("ERROR", f"Unexpected error during {call_label} API call: {e}")
        return None

def prettify_chart_spec(chart_spec: dict) -> dict:
    """ Applies styling to a Vega-Lite spec. """
    if not isinstance(chart_spec, dict): 
        return chart_spec
    new_spec = copy.deepcopy(chart_spec)
    new_spec["width"] = "container"
    new_spec["height"] = 300
    new_spec.setdefault("config", {})
    new_spec["config"]["range"] = { "category": [ "hsla(216, 81%, 50%, 1)", "hsla(47, 100%, 46%, 1)", "hsla(162, 53%, 55%, 1)"
                                                 , "hsla(351, 83%, 45%, 1)", "hsla(250, 88%, 65%, 1)", "hsla(25, 100%, 56%, 1)"
                                                 , "hsla(194, 100%, 72%, 1)", "hsla(284, 100%, 68%, 1)" ] 
                                }
    new_spec["config"]["axis"] = { "labelColor": "hsla(221, 18%, 44%, 1)",
                                   "titleColor": "hsla(221, 18%, 44%, 1)" 
                                }
    new_spec["config"]["legend"] = { "labelColor": "hsla(221, 18%, 44%, 1)", "titleColor": "hsla(221, 18%, 44%, 1)" }
    return new_spec

def display_non_text_content(content_parts: list, full_text_context: str | None = None, message_key_prefix: str = "msg"):
    """
    Renders charts, tables, related queries etc. from a list of content parts.
    Skips analyst1 tool results, fetched_table, search results as they are handled elsewhere or via citations.
    Optionally uses full_text_context to render citations/RQs.
    Removes st.info wrappers.
    """
    add_log("DEBUG", f"Displaying {len(content_parts)} non-text parts (in display_non_text_content).")
    analyst_text = ""
    all_citations_in_message = [] # For citation rendering

    # Pass 1: Extract data needed for context (citations, Analyst Text)
    for part in content_parts:
        content_type = part.get("type", "").lower()
        if content_type == "tool_results":
            tool_results_data = part.get('tool_results', {})
            tool_name = tool_results_data.get('name', 'N/A')
            try:
                inner_content_list = tool_results_data.get('content', [])
                if inner_content_list and isinstance(inner_content_list, list) and len(inner_content_list) > 0:
                    inner_item = inner_content_list[0]
                    inner_json = None
                    if isinstance(inner_item, dict): 
                        inner_json = inner_item.get('json', {})
                    elif isinstance(inner_item, str):
                        try: inner_json = json.loads(inner_item).get('json', {})
                        except json.JSONDecodeError: pass

                    if isinstance(inner_json, dict):
                        # Extract Citations from search results (even if name is empty)
                        # This populates all_citations_in_message for Pass 4
                        if "searchResults" in inner_json:
                            raw_citations = inner_json["searchResults"]
                            if isinstance(raw_citations, list):
                                for res in raw_citations:
                                    try: 
                                        num = int(res.get("source_id", -1))
                                        url = res.get("url"); text = res.get("text", "")
                                    except (ValueError, TypeError): 
                                        continue
                                    if num != -1: 
                                        all_citations_in_message.append({"number": num, "text": text, "url": url})
                        # Extract Text from analyst results
                        if tool_name == "analyst1":
                             if "text" in inner_json: 
                                 analyst_text += inner_json["text"] + "\n"
            except Exception as e: add_log("ERROR", f"Error extracting data from tool_results: {e}")

    # Pass 2: Render Analyst Interpretation Text
    if analyst_text:
        st.markdown(analyst_text)

    # Pass 3: Render other non-analyst parts (Charts, Other Tool Results)
    for i, part in enumerate(content_parts): # Use enumerate for unique key generation
        content_type = part.get("type", "").lower()
        tool_results_data = part.get('tool_results', {}) # Get tool results data for name check
        tool_name = tool_results_data.get('name', 'N/A')

        # Skip parts already handled or not meant for direct display here
        # Skip analyst1 results, text, tool_use, fetched_table, and search results
        if content_type in ["tool_use", "text", "fetched_table"] or tool_name == "analyst1" or tool_name == "search1" or part.get("tool_results", {}).get("name") == "search1": # Check both ways
             # Also check if it *contains* searchResults even if name is missing
             is_search_result = False
             if content_type == "tool_results":
                 try:
                     inner_json = tool_results_data.get('content', [{}])[0].get('json', {})
                     if isinstance(inner_json, dict) and "searchResults" in inner_json:
                         is_search_result = True
                 except Exception: pass # Ignore errors in check
             if is_search_result:
                 continue # Skip search results display here

        if content_type == "chart":
             chart_data = part.get("chart", {}); chart_spec = chart_data.get("chart_spec")
             if isinstance(chart_spec, str):
                 try: 
                     chart_spec = json.loads(chart_spec)
                 except json.JSONDecodeError: 
                     chart_spec = None
             if chart_spec and isinstance(chart_spec, dict):
                 try: 
                     st.vega_lite_chart(prettify_chart_spec(chart_spec), use_container_width=True)
                 except Exception as chart_err:
                     st.error(f"Chart Error: {chart_err}")
                     st.json(chart_spec)
             else: 
                st.warning("Invalid/missing chart spec.")
                st.json(chart_data) if chart_data else None
        elif content_type == "tool_results": # Display other tool results (NOT search, NOT analyst)
             if tool_name == "data_to_chart":
                 add_log("DEBUG", "Skipping display of raw data_to_chart tool result JSON.")
             else: # Fallback for truly unknown tool results
                 # We do not know what we got, so just skip it
                 add_log("WARN", f"Raw JSON for unhandled tool result: {tool_name}: {part}")
        else:
            # We do not know what we got, so just skip it 
            add_log("WARN", f"Unknown non-text content type: {content_type}: {part}")

    # Pass 4: Render Citations & Related Queries if context is available
    if full_text_context:
        # --- MODIFIED: Define extraction functions locally ---
        def extract_related_queries(text: str) -> list:
            queries = []
            if not text: 
                return queries
            try:
                # Use regex defined globally
                for match in re.finditer(RELATED_QUERIES_REGEX, text, re.DOTALL | re.IGNORECASE):
                    queries.append({"relatedQuery": match.group(1).strip(), "answer": match.group(2).strip()})
            except Exception as e: add_log("WARN", f"Error extracting RQs: {e}")
            return queries

        def extract_relevant_citations(text: str, all_citations: list) -> list:
            relevant_citations = []
            if not text or not all_citations: return relevant_citations
            try:
                # Use regex defined globally
                used_numbers = {int(num) for num in re.findall(CITATION_REGEX, text)}
                if used_numbers:
                    relevant_citations = [ cit for cit in all_citations if isinstance(cit.get("number"), int) and cit.get("number") in used_numbers ]
                    relevant_citations.sort(key=lambda x: x.get("number", 0))
            except Exception as e: add_log("WARN", f"Error extracting citations: {e}")
            return relevant_citations
        # --- End local function definitions ---

        related_queries = extract_related_queries(full_text_context)
        relevant_citations = extract_relevant_citations(full_text_context, all_citations_in_message)

        # --- MODIFIED: Display Related Queries using Expanders ---
        if related_queries:
             st.markdown("---") # Add separator
             st.subheader("Related Queries") # Add header
             for idx, rq in enumerate(related_queries):
                 with st.expander(rq['relatedQuery']): # Use query as title
                     st.markdown(rq['answer']) # Display answer inside

        if relevant_citations:
            st.markdown("---") # Add separator
            with st.container(border=True):
                st.write("Citations")
                for cit in relevant_citations:
                    url = cit.get('url')
                    with st.expander(f"Citation 【†{cit.get('number')}†】", expanded=False):
                        st.markdown(f"{cit.get('text', 'Source')}" + (f" ([link]({url}))" if url else ""))

    else:
         add_log("DEBUG", "Skipping Citation/RQ rendering as full text context was not provided.")


# --- Core Chat Logic ---
# --- Display existing chat messages ---
# Displays full user messages and assistant messages from history
st.markdown("---")
for i, message in enumerate(st.session_state.messages): # Use enumerate for unique key prefix
    with st.chat_message(message["role"]):
        full_text = ""
        non_text_parts = []
        sql_part = None
        table_part = None

        # Reconstruct full text and collect non-text parts from stored message
        for part in message.get("content", []):
            part_type = part.get("type")
            if part_type == "text":
                full_text += part.get("text", "")
            elif part_type == "fetched_table":
                table_part = part # Separate table part
            elif part_type == "tool_results" and part.get("tool_results", {}).get('name') == 'analyst1':
                 # Check if it contains SQL
                 try:
                     inner_content = part['tool_results'].get('content', [{}])[0]
                     inner_json = None
                     if isinstance(inner_content, dict): 
                         inner_json = inner_content.get('json', {})
                     elif isinstance(inner_content, str):
                         try: 
                             inner_json = json.loads(inner_content).get('json', {})
                         except json.JSONDecodeError: 
                             pass
                     if isinstance(inner_json, dict) and "sql" in inner_json:
                         sql_part = part # Separate SQL part
                     else:
                         non_text_parts.append(part) # Keep if not SQL
                 except Exception:
                     non_text_parts.append(part) # Keep if error parsing
            else:
                non_text_parts.append(part)

        # Display in desired order: Text -> SQL -> Table -> Others
        if full_text:
            # Display text from history (already filtered)
            st.markdown(full_text)
        if sql_part:
             try:
                 sql_to_view = sql_part['tool_results']['content'][0]['json']['sql']
                 with st.expander("View SQL"):
                     st.code(sql_to_view, language="sql")
             except Exception as e: add_log("WARN", f"Error displaying SQL from history: {e}")
        if table_part:
             with st.expander("View result table", expanded=False):
                 st.markdown(table_part.get("tableMarkdown"))
        if non_text_parts:
             # Pass the original full_text for context needed by citations/RQs
             # For history display, we might not have the *original* unfiltered text easily
             # Pass the filtered text for now, RQs/Citations might not render correctly from history view
             display_non_text_content(non_text_parts, full_text, message_key_prefix=f"msg_{i}")


# --- Handle User Input ---
if prompt := st.chat_input("What can I help with?"):
    # Append user message immediately to history for display
    user_message = { "role": "user", "content": [{"type": "text", "text": prompt}] }
    st.session_state.messages.append(user_message)
    # Rerun to display the user message and trigger processing below
    st.rerun()

# --- Process Turn: API Calls and Tool Execution (Reverted to Single Spinner / Placeholders) ---
# Check if the last message is from the user and needs processing
if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
    add_log("INFO", "Processing turn started...") # Add log entry here

    # Ensure session is available before proceeding
    if not session:
        st.error("Cannot process request: Snowflake connection is not established.")
        error_msg = "Sorry, I cannot process your request because the connection to Snowflake failed. Please check the configuration and logs."
        # Avoid adding duplicate error messages
        if not st.session_state.messages or st.session_state.messages[-1].get("role") != "assistant" or error_msg not in str(st.session_state.messages[-1].get("content",[])):
            st.session_state.messages.append({"role": "assistant", "content": [{"type": "text", "text": error_msg}]})
            st.rerun()
    else:
        # Get the latest messages to send to the API
        messages_to_send = st.session_state.messages

        # --- Assistant Response Area ---
        with st.chat_message("assistant"):
            # Initialize variables for this turn outside the spinner
            non_text_parts_1 = []
            non_text_parts_2 = []
            full_text_1_lst = [] # List to capture full text from generator 1
            full_text_2_lst = [] # List to capture full text from generator 2
            full_text_1 = "" # Unfiltered full text from stream 1
            full_text_2 = "" # Unfiltered full text from stream 2
            final_history_content = [] # Build the content list for the final message object
            sql_to_execute = None
            sql_dataframe_result = None
            query_id = None
            assistant_message_for_history = None # Initialize history message
            unique_non_text_to_display = [] # Initialize list for non-text display items
            sql_exec_success = False # Initialize flag
            api_call_2_success = False # Initialize flag

            # Use st.spinner for status indication during processing
            with st.spinner("Thinking..."):
                # === First API Call ===
                api_response_1 = call_agent_api(messages_to_send, call_label="1st")

            if api_response_1:
                # Stream text to UI, collect non-text parts, capture full text
                with st.spinner("Streaming..."):
                    try:
                        # Write stream directly, capture yielded text (potentially filtered by generator)
                        # The generator also populates non_text_parts_1 and full_text_1_lst via side effect
                        yielded_text_1 = st.write_stream(stream_text_and_collect_parts(api_response_1, full_text_1_lst, non_text_parts_1))
                        # Retrieve the full unfiltered text from the list populated by the generator
                        full_text_1 = full_text_1_lst[0] if full_text_1_lst else yielded_text_1 # Fallback to yielded text
                        add_log("DEBUG", f"Stream 1 finished. Full text length: {len(full_text_1)}, Collected non-text parts: {len(non_text_parts_1)}")
                    except Exception as e:
                            add_log("ERROR", f"Error during st.write_stream or processing stream 1: {e}")
                            # Update placeholder with error message
                            st.error(f"Error displaying response stream: {e}")
                            final_history_content = [{"type": "text", "text": f"Error processing response: {e}"}]

                # --- Process collected parts from Call 1 ---
                if not final_history_content: # Only proceed if no error during streaming
                    filtered_text_1 = re.sub(RELATED_QUERIES_REGEX, "", full_text_1, flags=re.DOTALL | re.IGNORECASE).strip()

                    # Build message content 1 (used for API call 2 context and potentially final history)
                    message_content_1 = []
                    if filtered_text_1: # Use filtered text
                        message_content_1.append({"type": "text", "text": filtered_text_1})
                    message_content_1.extend(non_text_parts_1) # Keep non-text parts as they were

                    # Check for SQL Execution Path
                    sql_tool_results_part = None # The specific tool_results part with SQL
                    for part in non_text_parts_1:
                        if part.get("type") == "tool_results" and part.get("tool_results", {}).get('name') == 'analyst1':
                                try:
                                    inner_content = part['tool_results'].get('content', [{}])[0]
                                    inner_json = None
                                    if isinstance(inner_content, dict): 
                                        inner_json = inner_content.get('json', {})
                                    elif isinstance(inner_content, str):
                                        try: inner_json = json.loads(inner_content).get('json', {})
                                        except json.JSONDecodeError: pass
                                    if isinstance(inner_json, dict) and "sql" in inner_json:
                                        sql_to_execute = inner_json["sql"]
                                        sql_tool_results_part = part # Store the part containing the SQL
                                        add_log("INFO", "SQL found for execution.")
                                        break # Found SQL
                                except Exception as e: add_log("WARN", f"Error checking tool_results for SQL: {e}")

                    # --- Execute SQL and potentially Call 2 (still inside spinner) ---
                    if sql_to_execute:
                        # Display SQL expander if SQL was generated
                        if sql_to_execute:
                            with st.expander("View SQL"):
                                st.code(sql_to_execute, language="sql")

                        with st.spinner("Executing SQL..."):
                            query_id, sql_dataframe_result = execute_sql(sql_to_execute)

                        if query_id:
                            # Display SQL results table immediately after execution (if successful)
                            if sql_dataframe_result is not None and not sql_dataframe_result.empty:
                                with st.expander("View result table", expanded=False):
                                    st.dataframe(sql_dataframe_result)
                            
                            sql_exec_success = True # Mark SQL success
                            add_log("INFO", f"SQL executed (Query ID: {query_id}). Making 2nd API call.")
                            first_assistant_message = {"role": "assistant", "content": message_content_1} # Send filtered context
                            tool_result_user_message = get_sql_exec_user_message(query_id)
                            messages_for_second_call = messages_to_send + [first_assistant_message, tool_result_user_message]

                            # === Second API Call ===
                            with st.spinner("Analyzing data.."):
                                api_response_2 = call_agent_api(messages_for_second_call, call_label="2nd")

                            if api_response_2:        
                                try:
                                    # Write the second stream directly, capture yielded text
                                    yielded_text_2 = st.write_stream(stream_text_and_collect_parts(api_response_2, full_text_2_lst, non_text_parts_2))
                                    # Retrieve full unfiltered text
                                    full_text_2 = full_text_2_lst[0] if full_text_2_lst else yielded_text_2
                                    add_log("DEBUG", f"Stream 2 finished. Full text length: {len(full_text_2)}, Collected non-text parts: {len(non_text_parts_2)}")
                                    api_call_2_success = True # Mark API 2 success
                                except Exception as e:
                                    add_log("ERROR", f"Error during st.write_stream or processing stream 2: {e}")
                                    # Update second placeholder with error
                                    st.error(f"Error displaying final response stream: {e}")
                                    # Append error to final_history_content which will be built later
                                    api_call_2_success = False # Ensure this is false on error
                                    final_history_content.append({"type": "text", "text": f"Error processing final response: {e}"})

                                # --- Construct final history content (SQL Path) ---
                                # This happens regardless of stream 2 success, but uses flags
                                # Rebuild history content based on success flags
                                if api_call_2_success:
                                    combined_final_content = []
                                    # Add Text 1 (FILTERED) if it exists
                                    if filtered_text_1: combined_final_content.append({"type": "text", "text": filtered_text_1})
                                    # Add Tool Use/Result from Call 1
                                    combined_final_content.extend([p for p in non_text_parts_1 if p.get("type") in ["tool_use", "tool_results"]])
                                    # Add Fetched Table
                                    if sql_dataframe_result is not None and not sql_dataframe_result.empty:
                                        combined_final_content.append({ "type": "fetched_table", "tableMarkdown": sql_dataframe_result.to_markdown(index=False), "toolResult": True })
                                    # Add Text 2 (FILTERED) if it exists
                                    if full_text_2:
                                        filtered_text_2 = re.sub(RELATED_QUERIES_REGEX, "", full_text_2, flags=re.DOTALL | re.IGNORECASE).strip()
                                        if filtered_text_2: # Only add if non-empty after filtering
                                            combined_final_content.append({"type": "text", "text": filtered_text_2})
                                    # Add other non-text parts from Call 2
                                    combined_final_content.extend([p for p in non_text_parts_2 if p.get("type") not in ["tool_use", "tool_results"]])

                                    final_history_content = combined_final_content # Set the final history content
                                # else: Error text already appended if stream 2 failed

                            else: # Second API call failed
                                # Add error directly to history content list (using message_content_1 which has filtered text 1)
                                error_text = "\n\nSorry, I could not process the results of the SQL query."
                                final_history_content = message_content_1 + [{"type": "text", "text": error_text}]
                                # Display error in the second placeholder
                                st.error(error_text.strip())


                        else: # SQL execution failed
                            # Add error directly to history content list (using message_content_1 which has filtered text 1)
                            error_text = "\n\nSorry, I encountered an error executing the required SQL."
                            final_history_content = message_content_1 + [{"type": "text", "text": error_text}]
                            # Display error in the first placeholder (since no second stream happened)
                            st.error(error_text.strip())
                    # End of SQL execution block
                    else: # No SQL to execute, single step path
                        final_history_content = message_content_1 # History is content from call 1 (with filtered text)

            else: # First API call failed
                # Error message already displayed in placeholder
                # final_history_content was already set
                pass

            # --- Spinner finishes here ---

            # --- Append the fully constructed message to history ---
            if final_history_content and any(final_history_content):
                 assistant_message_for_history = {"role": "assistant", "content": final_history_content}
                 # Check if the last message is identical to avoid duplicates from reruns/errors
                 # Compare content only, as role will be the same
                 if not st.session_state.messages or st.session_state.messages[-1].get("content") != assistant_message_for_history.get("content"):
                      st.session_state.messages.append(assistant_message_for_history)
                      add_log("DEBUG", f"Appended final constructed message to history.")
                 else:
                      add_log("DEBUG", "Skipping append to history, message content seems identical to last one.")
            else:
                 add_log("WARN", "No final message content generated, not appending to history.")


            # --- Display Final Non-Text Content (after spinner) ---
            # The streamed text is already visible in the placeholders above.
            # Now display the non-text elements directly using st commands.
            add_log("DEBUG", "Displaying final non-text content...")
            all_non_text_parts = non_text_parts_1 + non_text_parts_2
            seen_json_strings = set()
            unique_non_text_to_display = []
            for part in all_non_text_parts:
                try:
                    # Use separators=(',', ':') for compact representation, sort keys
                    json_string = json.dumps(part, sort_keys=True, separators=(',', ':'))
                    if json_string not in seen_json_strings:
                        seen_json_strings.add(json_string)
                        unique_non_text_to_display.append(part)
                except TypeError: # Handle unhashable types if json.dumps fails
                    # Basic check to avoid duplicates if adding anyway
                    is_duplicate = False
                    for existing_part in unique_non_text_to_display:
                         if part == existing_part: is_duplicate = True; break
                    if not is_duplicate: unique_non_text_to_display.append(part)

            # Combine *original* full text for context display for citations/RQs
            # Use the original full_text_1 and full_text_2 captured before filtering
            combined_full_text_context = (full_text_1 or "") + ("\n" + (full_text_2 or "") if sql_to_execute else "")
            # Display the unique non-text parts, providing text context
            # Exclude SQL tool result and fetched_table as they are handled explicitly elsewhere
            parts_for_display_func = [p for p in unique_non_text_to_display if not (p.get("type") == "tool_results" and p.get("tool_results", {}).get("name") == "analyst1") and p.get("type") != "fetched_table"]
            # Pass message index/key if needed for unique button keys
            # Using a simple counter for now within the display function call
            display_non_text_content(parts_for_display_func, combined_full_text_context, message_key_prefix=f"turn_{len(st.session_state.messages)}")

# --- Conditional Debug Display in Sidebar --- MOVED TO HERE ---
# Add check for debug_mode before accessing debug_log
if st.session_state.get("debug_mode", False):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Debug Log")
    with st.sidebar.container(height=250):
        log_container = st.empty()
        log_text = ""
        # Get logs safely, default to empty list
        current_logs = st.session_state.get("debug_log", [])
        for entry in reversed(current_logs): log_text += entry + "\n"
        # Use a unique key based on log length to force refresh
        log_container.text_area("Log Output", value=log_text, height=230, key=f"debug_log_display_{len(current_logs)}", disabled=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Message History (Debug)")
    with st.sidebar.expander("Show Message History JSON", expanded=False):
        st.json(st.session_state.messages)
