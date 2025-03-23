import streamlit as st
import pandas as pd
import base64
import requests
import time
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import folium
from streamlit_folium import st_folium
import hashlib
from datetime import timedelta

# Download VADER lexicon
nltk.download('vader_lexicon', quiet=True)

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Function to get real-time weather data
def get_weather(city):
    API_KEY = "7a2bc514b43fc8dc820ae1566673cf29"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
    try:
        response = requests.get(url).json()
        if response.get("cod") == 200:
            weather = {
                "Temperature": f"{response['main']['temp']}°C",
                "Condition": response["weather"][0]["description"].capitalize(),
                "Humidity": f"{response['main']['humidity']}%",
                "Wind Speed": f"{response['wind']['speed']} m/s"
            }
            # Get weather icon
            icon_code = response["weather"][0]["icon"]
            weather_icon = f"https://openweathermap.org/img/wn/{icon_code}@2x.png"
            return weather, weather_icon
        else:
            return {"Error": "Weather data not available"}, None
    except Exception as e:
        return {"Error": f"Failed to fetch data: {str(e)}"}, None

# Function to encode image in base64
def get_base64_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()
    except FileNotFoundError:
        st.error(f"Error: File not found at {image_path}")
        return ""

# Function to load dataset
@st.cache_data
def load_data():
    try:
        file_path = "Dataset.xlsx"
        df = pd.read_excel(file_path)
        # Ensure sentiment_score is numeric
        df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce")
        # Ensure Price Column is Numeric
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Function to get hotel coordinates using a geocoding API
@st.cache_data
def get_hotel_coordinates(hotel_name, city_name):
    try:
        # Use Nominatim (OpenStreetMap) geocoding service
        base_url = "https://nominatim.openstreetmap.org/search"
        search_term = f"{hotel_name}, {city_name}"
        params = {
            "q": search_term,
            "format": "json",
            "limit": 1,
        }
        
        # Add a user agent to comply with Nominatim's usage policy
        headers = {
            "User-Agent": "SmartAccommodationFinderApp/1.0"
        }
        
        response = requests.get(base_url, params=params, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                return [float(data[0]["lat"]), float(data[0]["lon"])]
        
        # Fallback to city center coordinates with small random offsets if geocoding fails
        return fallback_coordinates(hotel_name, city_name)
    
    except Exception as e:
        st.warning(f"Could not geocode {hotel_name} in {city_name}: {str(e)}")
        return fallback_coordinates(hotel_name, city_name)

# Fallback function to use when geocoding fails
def fallback_coordinates(hotel_name, city_name):
    # City base coordinates
    city_coordinates = {
        "Manali": [32.2396, 77.1887],
        "Darjeeling": [27.0410, 88.2663],
        "Munnar": [10.0889, 77.0595]
    }
    center_lat, center_lon = city_coordinates.get(city_name, [20.5937, 78.9629])
    
    # Add a small random offset (but cache it based on hotel name for consistency)
    hash_obj = hashlib.md5(hotel_name.encode())
    hash_hex = hash_obj.hexdigest()
    
    # Use the hash to create small offsets (between -0.002 and 0.002)
    offset_multiplier = int(hash_hex[:8], 16) / (16**8)
    lat_offset = (offset_multiplier * 0.004) - 0.002
    
    offset_multiplier = int(hash_hex[8:16], 16) / (16**8)
    lon_offset = (offset_multiplier * 0.004) - 0.002
    
    return [center_lat + lat_offset, center_lon + lon_offset]

# Function to create Folium map for a single hotel (cached to prevent regeneration)
@st.cache_data
def create_single_hotel_map(hotel_name, hotel_price, hotel_ratings, sentiment_score, city_name):
    # Create a base map centered on the city
    city_coordinates = {
        "Manali": [32.2396, 77.1887],
        "Darjeeling": [27.0410, 88.2663],
        "Munnar": [10.0889, 77.0595]
    }
    center_lat, center_lon = city_coordinates.get(city_name, [20.5937, 78.9629])
    
    # Get more accurate coordinates for this hotel
    hotel_lat, hotel_lon = get_hotel_coordinates(hotel_name, city_name)
    
    # Center the map on the hotel instead of the city center
    m = folium.Map(location=[hotel_lat, hotel_lon], zoom_start=15, 
                   tiles="CartoDB positron")
    
    # Use sentiment score to determine marker color
    if sentiment_score >= 0.75:
        color = 'green'
    elif sentiment_score >= 0.5:
        color = 'blue'
    else:
        color = 'orange'
    
    # Create a popup with hotel info
    popup_html = f"""
    <div style="width: 250px; font-family: 'Arial', sans-serif;">
        <h4 style="color: #2C3E50; margin-bottom: 10px;">{hotel_name}</h4>
        <p style="margin: 5px 0;"><b style="color: #2980B9;">Price:</b> ₹{hotel_price:,}</p>
        <p style="margin: 5px 0;"><b style="color: #2980B9;">Rating:</b> {hotel_ratings}</p>
        <p style="margin: 5px 0;"><b style="color: #2980B9;">Sentiment:</b> {sentiment_score:.2f}</p>
    </div>
    """
    
    # Add a marker with custom icon and popup
    folium.Marker(
        location=[hotel_lat, hotel_lon],
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=hotel_name,
        icon=folium.Icon(color=color, icon="hotel", prefix="fa")
    ).add_to(m)
    
    return m

# Generate sentiment description based on score
def get_sentiment_description(score):
    if score >= 0.8:
        return "Excellent", "Guests absolutely love this hotel!"
    elif score >= 0.7:
        return "Very Good", "Very positive guest experiences reported"
    elif score >= 0.6:
        return "Good", "Guests generally have positive experiences"
    elif score >= 0.5:
        return "Satisfactory", "Mixed reviews with positive experiences"
    else:
        return "Mixed", "Some guests had concerns about their stay"

# Function to get amenity icons
def get_amenity_icon(amenity):
    amenity = amenity.strip().lower()
    icons = {
        "wifi": "wifi",
        "swimming pool": "swimming-pool",
        "pool": "swimming-pool",
        "parking": "parking",
        "spa": "spa",
        "gym": "dumbbell",
        "fitness": "dumbbell",
        "restaurant": "utensils",
        "bar": "glass-martini",
        "breakfast": "coffee",
        "air conditioning": "snowflake",
        "air conditioner": "snowflake",
        "ac": "snowflake",
        "pet friendly": "paw",
        "beach": "umbrella-beach",
        "room service": "concierge-bell",
        "laundry": "tshirt",
        "tv": "tv",
        "balcony": "door-open"
    }
    
    for key, icon in icons.items():
        if key in amenity:
            return icon
    
    return "check-circle"  # Default icon

# Custom card for hotel display
def create_hotel_card(hotel, rank=None):
    name = hotel['Hotel Name']
    price = hotel['Price']
    rating = hotel['Ratings']
    sentiment = hotel['sentiment_score']
    destination = hotel['Destination']
    
    # Handle different formats of amenities
    if isinstance(hotel['Amenities'], str):
        amenities = hotel['Amenities'].split(',')
    elif isinstance(hotel['Amenities'], list):
        amenities = hotel['Amenities']
    else:
        amenities = []
    
    sentiment_label, sentiment_desc = get_sentiment_description(sentiment)
    
    # Card header with rank
    rank_emoji = ""
    if rank == 0:
        rank_emoji = "🥇 "
    elif rank == 1:
        rank_emoji = "🥈 "
    elif rank == 2:
        rank_emoji = "🥉 "
    
    # HTML/CSS for styled card
    card_html = f"""
    <div style="background-color: white; border-radius: 10px; padding: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
        <h3 style="color: #1E3A8A; margin-top: 0;">{rank_emoji}{name}</h3>
        <div style="display: flex; margin-bottom: 10px;">
            <div style="background-color: #3B82F6; color: white; font-weight: bold; padding: 3px 10px; border-radius: 15px; margin-right: 10px;">
                ₹{price:,}
            </div>
            <div style="background-color: #10B981; color: white; font-weight: bold; padding: 3px 10px; border-radius: 15px; margin-right: 10px;">
                ⭐ {rating}
            </div>
            <div style="background-color: #8B5CF6; color: white; font-weight: bold; padding: 3px 10px; border-radius: 15px;">
                {sentiment_label} ({sentiment:.2f})
            </div>
        </div>
        <p style="color: #4B5563; margin-bottom: 5px;"><i class="fas fa-map-marker-alt"></i> {destination}</p>
        <p style="color: #4B5563; font-style: italic; margin-bottom: 15px;">{sentiment_desc}</p>
    """
    
    
    
    # Close the HTML
    card_html += """
    </div>
    """
    
    return card_html

# Calculate length of stay and return formatted nights text
def calculate_stay_length(check_in, check_out):
    delta = check_out - check_in
    nights = delta.days
    if nights == 1:
        return "1 night"
    else:
        return f"{nights} nights"

# Page configuration
st.set_page_config(
    page_title="Yatri Mitra", 
    page_icon="", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data
df = load_data()

if df.empty:
    st.error("Failed to load dataset. Please check the file path.")
    st.stop()

# Encode background image
bg_image_path = "background.jpg"
bg_image_base64 = get_base64_image(bg_image_path)

# Inject custom CSS for styling
st.markdown(
    """
    <style>
    @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css');
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
    }
    
    .stApp {
        background-color: #f0f2f6;
    }
    
    .header-container {
        background-color: #1E3A8A;
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
        background-image: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%);
    }
    
    .subheader {
        color: #1E3A8A;
        font-weight: bold;
        border-left: 5px solid #3B82F6;
        padding-left: 10px;
        margin: 25px 0 15px 0;
    }
    
    .search-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .results-container {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    
    .weather-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
        height: 100%;
    }
    
    .map-container {
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .hotel-data-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border-radius: 10px;
        border: 1px solid #E5E7EB;
        background-color: #F9FAFB;
    }
    
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        height: 100%;
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    
    /* Button styling */
    .search-button {
        background-color: #10B981;
        color: white;
        padding: 10px 20px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        cursor: pointer;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }
    
    .search-button:hover {
        background-color: #059669;
    }
    
    /* Hiding Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# If background image is present, add it to the CSS
if bg_image_base64:
    st.markdown(
        f"""
        <style>
        .header-container {{
            background-image: url("data:image/jpeg;base64,{bg_image_base64}");
            background-size: cover;
            background-position: center;
            position: relative;
        }}
        .header-container::before {{
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            bottom: 0;
            left: 0;
            border-radius: 10px;
        }}
        .header-content {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Header
st.markdown(
    """
    <div class="header-container">
        <div class="header-content">
            <h1>Yatri Mitra - Smart Accmodation Finder</h1>
            <p>Discover the perfect stay for your next adventure with personalized recommendations</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Main content
with st.container():
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    # Search form header
    st.markdown('<h2 class="subheader">Find Your Perfect Stay</h2>', unsafe_allow_html=True)
    
    # Form layout
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        destination = st.selectbox("🌍 Destination", df["Destination"].unique(), 
                                   help="Select your travel destination")
        
        # Simplified price range selection with a single control
        price_options = ["Budget (₹0 - ₹5000)", "Comfort (₹5001 - ₹10000)", "Luxury (₹10001+)"]
        price_selection = st.radio("💰 Price Range", price_options, 
                                 help="Select your budget preference")
        
        # Convert selection to actual price range for filtering
        if price_selection == "Budget (₹0 - ₹5000)":
            price_range = "₹0 - ₹5000"
        elif price_selection == "Comfort (₹5001 - ₹10000)":
            price_range = "₹5001 - ₹10000"
        else:
            price_range = "₹10001 and above"
    
    with col2:
        # Date selection
        today = pd.Timestamp.now().date()
        tomorrow = today + timedelta(days=1)
        three_days = today + timedelta(days=3)
        
        checkin_date = st.date_input("📅 Check-in Date", value=tomorrow, 
                                     min_value=today, 
                                     help="Select your arrival date")
        
        checkout_date = st.date_input("📅 Check-out Date", value=three_days, 
                                      min_value=checkin_date + timedelta(days=1), 
                                      help="Select your departure date")
        
        # Calculate and display stay length
        stay_length = calculate_stay_length(checkin_date, checkout_date)
        st.markdown(f"<p><i>Stay duration: {stay_length}</i></p>", unsafe_allow_html=True)
    
    with col3:
        # Amenities selection
        all_amenities = sorted(set(",".join(df["Amenities"].dropna()).split(",")))
        all_amenities = [amenity.strip() for amenity in all_amenities if amenity.strip()]
        selected_amenities = st.multiselect("🛎 Must-have Amenities", 
                                          all_amenities, 
                                          help="Select amenities you need during your stay")
        
        # Add a search button
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔍 Find Hotels", type="primary"):
            st.balloons()

    
    st.markdown('</div>', unsafe_allow_html=True)

# Weather information
st.markdown('<div class="results-container">', unsafe_allow_html=True)
st.markdown(f'<h2 class="subheader">Current Weather in {destination}</h2>', unsafe_allow_html=True)

weather_data, weather_icon = get_weather(destination)
weather_cols = st.columns(4)

for i, (key, value) in enumerate(weather_data.items()):
    with weather_cols[i % 4]:
        st.markdown(
            f"""
            <div class="weather-card">
                <h3 style="color: #3B82F6; margin-top: 0;">{key}</h3>
                <p style="font-size: 1.5rem; font-weight: bold;">{value}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

if weather_icon:
    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 10px;">
            <img src="{weather_icon}" alt="Weather Icon" style="width: 60px; height: 60px;">
        </div>
        """,
        unsafe_allow_html=True
    )
st.markdown('</div>', unsafe_allow_html=True)

# Filter hotels based on criteria
if price_range == "₹0 - ₹5000":
    filtered_df = df[(df["Destination"] == destination) & (df["Price"] <= 5000)]
elif price_range == "₹5001 - ₹10000":
    filtered_df = df[(df["Destination"] == destination) & (df["Price"].between(5001, 10000, inclusive="both"))]
else:
    filtered_df = df[(df["Destination"] == destination) & (df["Price"] > 10000)]

# Filter based on selected amenities
if selected_amenities:
    filtered_df = filtered_df[filtered_df["Amenities"].apply(lambda x: all(amenity in str(x).split(",") for amenity in selected_amenities))]

# Get the top 3 hotels with highest sentiment scores
if "sentiment_score" not in filtered_df.columns:
    st.error("⚠ 'sentiment_score' column not found in dataset.")
else:
    top_hotels = filtered_df.dropna(subset=["sentiment_score"]).sort_values(by="sentiment_score", ascending=False).head(3)

    if top_hotels.empty:
        st.warning("⚠ No hotels found for this destination and criteria. Try adjusting your filters!")
    else:
        # Top hotels section
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.markdown('<h2 class="subheader">Top Recommended Hotels</h2>', unsafe_allow_html=True)
        st.markdown(f"<p>Found <b>{len(filtered_df)}</b> hotels matching your criteria. Here are our top picks:</p>", unsafe_allow_html=True)
        
        # Display each hotel with improved styling
        for i, (idx, hotel) in enumerate(top_hotels.iterrows()):
            hotel_name = hotel['Hotel Name']
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Display hotel card
                st.markdown(create_hotel_card(hotel, rank=i), unsafe_allow_html=True)
                
                # Create a link to Google Maps
                hotel_name_encoded = hotel_name.replace(' ', '+')
                location_encoded = hotel['Destination'].replace(' ', '+')
                st.markdown(f"[🗺 View on Google Maps](https://www.google.com/maps/search/?api=1&query={hotel_name_encoded}+{location_encoded})")
                
            with col2:
                # Create a styled container for the map
                st.markdown('<div class="map-container">', unsafe_allow_html=True)
                # Create an individual map for this hotel
                hotel_map = create_single_hotel_map(
                    hotel_name, 
                    hotel['Price'], 
                    hotel['Ratings'], 
                    hotel['sentiment_score'], 
                    destination
                )
                
                # Use a unique key for each folium map
                map_key = f"map_{hotel_name}_{i}"
                st_folium(hotel_map, width=600, height=300, key=map_key)
                st.markdown('</div>', unsafe_allow_html=True)
            
            
            # Add a separator between hotels
            if i < len(top_hotels) - 1:
                st.markdown("<hr style='margin: 30px 0; border-color: #E5E7EB;'>", unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)