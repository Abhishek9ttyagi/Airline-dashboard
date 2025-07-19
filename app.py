import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests
import pandas as pd
import re
from datetime import date, timedelta
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURATION ---
# API keys are now loaded from the .env file for better security.
AMADEUS_API_KEY = os.getenv("AMADEUS_API_KEY")
AMADEUS_API_SECRET = os.getenv("AMADEUS_API_SECRET")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Define cities globally to be accessible by all routes
CITIES = {'SYD': 'Sydney', 'MEL': 'Melbourne', 'BNE': 'Brisbane', 'PER': 'Perth', 'ADL': 'Adelaide', 'CBR': 'Canberra', 'HBA': 'Hobart'}
# --- FLASK APP INITIALIZATION ---
app = Flask(__name__)

if GEMINI_API_KEY and GEMINI_API_KEY != "YOUR_GEMINI_API_KEY":
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
else:
    print("Warning: Gemini API Key is not set in app.py. AI insights will be disabled.")

# --- HELPER FUNCTIONS ---

def markdown_to_html(text):
    """
    Converts a markdown-like text from Gemini into simple HTML.
    - Converts **bold** to <strong>
    - Converts * or - list items to <ul><li>...</li></ul>
    - Wraps standalone lines in <p> tags.
    """
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    lines = text.strip().split('\n')
    html_output = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Check for list items
        if line.startswith('* ') or line.startswith('- '):
            if not in_list:
                html_output.append('<ul>')
                in_list = True
            html_output.append(f'<li>{line[2:]}</li>')
        else:
            if in_list:
                html_output.append('</ul>')
                in_list = False
            html_output.append(f'<p>{line}</p>')
            
    if in_list:
        html_output.append('</ul>')
        
    return ''.join(html_output)

def get_amadeus_access_token():
    """
    Fetches a new access token from the Amadeus API using client credentials.
    This token is required to make authorized API calls.
    """
    url = "https://test.api.amadeus.com/v1/security/oauth2/token"
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    data = f'grant_type=client_credentials&client_id={AMADEUS_API_KEY}&client_secret={AMADEUS_API_SECRET}'
    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
        return response.json().get('access_token')
    except requests.exceptions.RequestException as e:
        print(f"Error getting Amadeus token: {e}")
        return None

def search_flight_deals(access_token, origin, destination, departure_date):
    """
    Searches for the cheapest one-way flight offers for a specific date using the Amadeus API.
    """
    url = "https://test.api.amadeus.com/v2/shopping/flight-offers"
    
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {
        'originLocationCode': origin,
        'destinationLocationCode': destination,
        'departureDate': departure_date,
        'adults': '1',
        'nonStop': 'true', # We only want direct flights for simpler analysis
        'currencyCode': 'AUD',
        'max': 50  # Get up to 50 offers to analyze price variations
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get('data', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching flight data for {origin}->{destination}: {e}")
        return []

def process_flight_data(flight_data, origin, destination):
    """
    Cleans and structures the raw JSON data from Amadeus into a pandas DataFrame.
    """
    if not flight_data:
        return pd.DataFrame()

    processed_list = [{'origin': origin, 'destination': destination, 'price': float(offer['price']['total']), 'airline': offer['validatingAirlineCodes'][0], 'departure_date': offer['itineraries'][0]['segments'][0]['departure']['at'].split('T')[0]} for offer in flight_data]
    
    return pd.DataFrame(processed_list)

def get_ai_insights(df):
    """
     Generates insights from the data using the Google Gemini API.
    """

    # Check if the API key is available before trying to use the API
    if not GEMINI_API_KEY or GEMINI_API_KEY == "YOUR_GEMINI_API_KEY":
        return "<p>Gemini API key not configured. Insights are unavailable.</p>"
    if df.empty:
        return "<p>No flight data was available to generate insights.</p>"

    # Select the Gemini model
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # Create a prompt that gives the AI context about its role
    prompt = f"""
    You are a market analyst for a chain of hostels in Australia.
    Your goal is to help the marketing team understand airline booking trends to better target customers.
    Based on the following summary of average flight prices from various cities, provide actionable insights.

    Data Summary (Average one-way price in AUD to the destination):
    {df.to_string()}

    Please provide a brief summary as a list of key takeaways. Use bullet points (starting with '*') for list items.
    For the "General advice" section, please make "**General Marketing Advice:**" a standalone bolded line, followed by a separate bulleted list of the specific advice points.

    Format your response using simple markdown. Use bullet points (starting with '*') for lists and bold text for emphasis. Do not use headings.
    """
    try:
        response = model.generate_content(prompt)
        # It's good practice to check if the response was blocked
        if not response.parts:
            return "<p>The AI response was blocked for safety reasons. Please try a different query.</p>"
        
        return markdown_to_html(response.text)
    except Exception as e:
        print(f"Error getting Gemini insights: {e}")
        return "<p>Could not generate AI insights. The Gemini API might be unavailable or the key is invalid.</p>"

# --- FLASK WEB ROUTES ---

@app.route('/', methods=['GET', 'POST'])
def index():
    # On form submission, get the selected city. Default to Sydney on first load.
    destination_city = request.form.get('destination', 'SYD')

    # We will analyze demand INTO the destination from all other cities in our list.
    origin_cities = [code for code in CITIES if code != destination_city]
    
    access_token = get_amadeus_access_token()
    if not access_token:
        # Display an error if we can't connect to the data source
        return "Error: Could not authenticate with Amadeus API. Please check your API credentials in app.py.", 500

    # --- Data Fetching for Popular Routes Chart (Snapshot for tomorrow) ---
    popular_routes_df = pd.DataFrame()
    snapshot_date = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    for origin in origin_cities:
        flight_data = search_flight_deals(access_token, origin, destination_city, snapshot_date)
        df = process_flight_data(flight_data, origin, destination_city)
        if not df.empty:
            popular_routes_df = pd.concat([popular_routes_df, df], ignore_index=True)

    # --- Data Fetching for Price Trend Chart (30-day trend for one representative route) ---
    price_trends_df = pd.DataFrame()
    # Use the first origin city as a representative route for the trend analysis to be efficient
    trend_origin_city = origin_cities[0] if origin_cities else None
    if trend_origin_city:
        for i in range(1, 31): # Fetch data for the next 30 days
            trend_date = (date.today() + timedelta(days=i)).strftime("%Y-%m-%d")
            flight_data = search_flight_deals(access_token, trend_origin_city, destination_city, trend_date)
            if flight_data:
                df = process_flight_data(flight_data, trend_origin_city, destination_city)
                if not df.empty:
                    # For the trend, we only need the cheapest price for that day to keep the line clean
                    min_price_df = df.loc[df.groupby('departure_date')['price'].idxmin()]
                    price_trends_df = pd.concat([price_trends_df, min_price_df], ignore_index=True)

    # --- Prepare Data for Charts and AI Summary ---
    price_trends_data = {}
    popular_routes_data = {}
    ai_summary = "No data found for the selected destination. This may be due to the route having no availability in the next month or an API error."
    trend_chart_title = "Daily Price Trend (Next Month)"

    if not price_trends_df.empty:
        # Aggregate data for the "Price Trends" chart (average price per day)
        price_trends = price_trends_df.groupby('departure_date')['price'].mean().round(2).sort_index()
        price_trends_data = {
            'labels': price_trends.index.tolist(),
            'data': price_trends.values.tolist()
        }
        trend_chart_title = f"Daily Price Trend: {CITIES.get(trend_origin_city)} to {CITIES.get(destination_city)}"
        
    if not popular_routes_df.empty:
        # Aggregate data for the "Popular Routes" chart (average price per route)
        popular_routes = popular_routes_df.groupby('origin')['price'].mean().sort_values().round(2)
        popular_routes_data = {
            'labels': [f"{CITIES.get(origin, origin)} -> {CITIES.get(destination_city, destination_city)}" for origin in popular_routes.index],
            'data': popular_routes.values.tolist(),
            'origin_codes': popular_routes.index.tolist() # Add origin codes for the frontend
        }
        # Generate the AI summary
        ai_summary = get_ai_insights(popular_routes.reset_index().rename(columns={'origin': 'Route from', 'price': 'Average Price (AUD)'}))

    # Render the webpage, passing all the processed data to the template
    return render_template('index.html',
                           cities=CITIES,
                           selected_destination=destination_city,
                           price_trends=price_trends_data,
                           popular_routes=popular_routes_data,
                           ai_summary=ai_summary,
                           trend_chart_title=trend_chart_title)

@app.route('/api/price-trend')
def api_price_trend():
    """API endpoint to fetch price trend data for a specific route."""
    origin = request.args.get('origin')
    destination = request.args.get('destination')
    
    if not origin or not destination:
        return jsonify({'error': 'Missing origin or destination parameters'}), 400

    access_token = get_amadeus_access_token()
    if not access_token:
        return jsonify({'error': 'Could not authenticate with Amadeus API'}), 500

    price_trends_df = pd.DataFrame()
    for i in range(1, 31): # Fetch data for the next 30 days
        trend_date = (date.today() + timedelta(days=i)).strftime("%Y-%m-%d")
        flight_data = search_flight_deals(access_token, origin, destination, trend_date)
        if flight_data:
            df = process_flight_data(flight_data, origin, destination)
            if not df.empty:
                min_price_df = df.loc[df.groupby('departure_date')['price'].idxmin()]
                price_trends_df = pd.concat([price_trends_df, min_price_df], ignore_index=True)

    if price_trends_df.empty:
        return jsonify({'labels': [], 'data': [], 'chart_title': f'No Trend Data for {origin} to {destination}'})

    price_trends = price_trends_df.groupby('departure_date')['price'].mean().round(2).sort_index()
    response_data = {
        'labels': price_trends.index.tolist(),
        'data': price_trends.values.tolist(),
        'chart_title': f"Daily Price Trend: {CITIES.get(origin)} to {CITIES.get(destination)}"
    }
    return jsonify(response_data)

if __name__ == '__main__':
    # Run the Flask app in debug mode for development
    app.run(debug=True)