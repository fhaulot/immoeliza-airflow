"""
Streamlit Dashboard for Belgian Real Estate Price Prediction
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from pathlib import Path

# Configuration
st.set_page_config(
    page_title="Belgian Real Estate Predictor",
    page_icon="🏡",
    layout="wide"
)

API_URL = "http://localhost:8000"

# Load data for analysis
@st.cache_data
def load_data():
    """Load the scraped data for analysis."""
    # Try different possible locations
    possible_paths = [
        Path(__file__).parent.parent / "data" / "immovlan_scraped_data.csv",
        Path(__file__).parent.parent / "immovlan_scraped_data_test.csv",
        Path(__file__).parent.parent / "immovlan_scraped_data.csv",
        Path("/app/data/immovlan_scraped_data.csv"),
        Path("/app/immovlan_scraped_data_test.csv"),
    ]
    
    for data_path in possible_paths:
        if data_path.exists():
            return pd.read_csv(data_path)
    
    # Fallback to analysis data
    analysis_path = Path(__file__).parent.parent / "analyse" / "processed_for_analysis.csv"
    if analysis_path.exists():
        return pd.read_csv(analysis_path)
    
    return None

# Main title
st.title("🏡 Belgian Real Estate Price Predictor")
st.markdown("---")

# Sidebar - Prediction Form
with st.sidebar:
    st.header("Property Details")
    
    property_type = st.selectbox("Property Type", ["HOUSE", "APARTMENT"])
    subtype = st.selectbox("Subtype", ["HOUSE", "APARTMENT", "VILLA", "TOWNHOUSE"])
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    surface = st.number_input("Habitable Surface (m²)", min_value=20, max_value=1000, value=150)
    
    st.subheader("Location")
    postal_code = st.number_input("Postal Code", min_value=1000, max_value=9999, value=1000)
    province = st.selectbox(
        "Province",
        ["Brussels", "Antwerp", "Flemish Brabant", "Walloon Brabant",
         "West Flanders", "East Flanders", "Hainaut", "Liège",
         "Limburg", "Luxembourg", "Namur"]
    )
    region = st.selectbox("Region", ["Brussels", "Flanders", "Wallonia"])
    
    st.subheader("Features")
    has_garden = st.checkbox("Garden")
    garden_surface = 0
    if has_garden:
        garden_surface = st.number_input("Garden Surface (m²)", min_value=0, max_value=5000, value=50)
    has_terrace = st.checkbox("Terrace")
    has_parking = st.checkbox("Parking")
    
    st.subheader("Condition")
    building_condition = st.selectbox(
        "Building Condition",
        ["NEW", "GOOD", "TO_RENOVATE", "TO_RESTORE", "UNKNOWN"]
    )
    epc_score = st.selectbox(
        "EPC Score",
        ["A++", "A+", "A", "B", "C", "D", "E", "F", "G", "UNKNOWN"]
    )
    
    predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)

# Main content area
tab1, tab2, tab3 = st.tabs(["📊 Price Prediction", "🗺️ Price by Province", "📍 Top Postal Codes"])

# Tab 1: Price Prediction
with tab1:
    if predict_button:
        with st.spinner("Predicting price..."):
            # Prepare data for API
            data = {
                "type": property_type,
                "subtype": subtype,
                "bedroomCount": bedrooms,
                "habitableSurface": surface,
                "postCode": postal_code,
                "province": province,
                "region": region,
                "hasGarden": has_garden,
                "gardenSurface": garden_surface,
                "hasTerrace": has_terrace,
                "hasParking": has_parking,
                "buildingCondition": building_condition,
                "epcScore": epc_score
            }
            
            try:
                response = requests.post(f"{API_URL}/predict", json=data, timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_price = result["predicted_price"]
                    
                    # Display result
                    st.success("Prediction successful!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Predicted Price", f"€ {predicted_price:,.0f}")
                    
                    with col2:
                        price_per_sqm = predicted_price / surface
                        st.metric("Price per m²", f"€ {price_per_sqm:,.0f}")
                    
                    with col3:
                        if has_garden and garden_surface > 0:
                            total_surface = surface + garden_surface
                            st.metric("Total Surface", f"{total_surface:,.0f} m²")
                        else:
                            st.metric("Habitable Surface", f"{surface:,.0f} m²")
                    
                    # Property details
                    st.subheader("Property Details")
                    details_df = pd.DataFrame({
                        "Feature": ["Type", "Bedrooms", "Surface", "Location", "Condition", "EPC"],
                        "Value": [
                            f"{property_type} - {subtype}",
                            bedrooms,
                            f"{surface} m²",
                            f"{postal_code} - {province}",
                            building_condition,
                            epc_score
                        ]
                    })
                    st.dataframe(details_df, hide_index=True, use_container_width=True)
                    
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"Error connecting to API: {str(e)}")
                st.info("Make sure the API is running on http://localhost:8000")
    else:
        st.info("👈 Fill in the property details and click 'Predict Price' to get an estimate")
        
        # Show example
        st.subheader("How it works")
        st.markdown("""
        1. **Enter property details** in the sidebar
        2. **Click 'Predict Price'** to get an instant estimate
        3. **View analysis** in other tabs for market insights
        """)

# Tab 2: Price by Province
with tab2:
    st.header("Average Price by Province")
    
    df = load_data()
    
    if df is not None and 'price' in df.columns and 'province' in df.columns:
        # Calculate average price by province
        province_stats = df.groupby('province').agg({
            'price': ['mean', 'median', 'count']
        }).round(0)
        
        province_stats.columns = ['Average Price', 'Median Price', 'Count']
        province_stats = province_stats.reset_index()
        province_stats = province_stats.sort_values('Average Price', ascending=False)
        
        # Create bar chart
        fig = px.bar(
            province_stats,
            x='province',
            y='Average Price',
            title='Average Property Price by Province',
            labels={'province': 'Province', 'Average Price': 'Average Price (€)'},
            color='Average Price',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show statistics table
        st.subheader("Province Statistics")
        province_stats['Average Price'] = province_stats['Average Price'].apply(lambda x: f"€ {x:,.0f}")
        province_stats['Median Price'] = province_stats['Median Price'].apply(lambda x: f"€ {x:,.0f}")
        st.dataframe(province_stats, hide_index=True, use_container_width=True)
    else:
        st.warning("No data available. Please run the scraper first.")

# Tab 3: Top Postal Codes
with tab3:
    st.header("Top Postal Codes")
    
    df = load_data()
    
    if df is not None and 'price' in df.columns and 'postCode' in df.columns:
        # Calculate average price by postal code
        postal_stats = df.groupby('postCode').agg({
            'price': ['mean', 'count']
        }).round(0)
        
        postal_stats.columns = ['Average Price', 'Count']
        postal_stats = postal_stats.reset_index()
        
        # Filter postal codes with at least 1 property
        postal_stats = postal_stats[postal_stats['Count'] >= 1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔝 Top 10 Most Expensive")
            top_10_expensive = postal_stats.nlargest(10, 'Average Price')
            
            fig_expensive = px.bar(
                top_10_expensive,
                x='postCode',
                y='Average Price',
                title='Most Expensive Postal Codes',
                labels={'postCode': 'Postal Code', 'Average Price': 'Average Price (€)'},
                color='Average Price',
                color_continuous_scale='Reds'
            )
            fig_expensive.update_layout(height=400)
            st.plotly_chart(fig_expensive, use_container_width=True)
            
            # Table
            top_10_expensive_display = top_10_expensive.copy()
            top_10_expensive_display['Average Price'] = top_10_expensive_display['Average Price'].apply(lambda x: f"€ {x:,.0f}")
            st.dataframe(top_10_expensive_display, hide_index=True, use_container_width=True)
        
        with col2:
            st.subheader("💰 Top 10 Most Affordable")
            top_10_affordable = postal_stats.nsmallest(10, 'Average Price')
            
            fig_affordable = px.bar(
                top_10_affordable,
                x='postCode',
                y='Average Price',
                title='Most Affordable Postal Codes',
                labels={'postCode': 'Postal Code', 'Average Price': 'Average Price (€)'},
                color='Average Price',
                color_continuous_scale='Greens'
            )
            fig_affordable.update_layout(height=400)
            st.plotly_chart(fig_affordable, use_container_width=True)
            
            # Table
            top_10_affordable_display = top_10_affordable.copy()
            top_10_affordable_display['Average Price'] = top_10_affordable_display['Average Price'].apply(lambda x: f"€ {x:,.0f}")
            st.dataframe(top_10_affordable_display, hide_index=True, use_container_width=True)
    else:
        st.warning("No data available. Please run the scraper first.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Powered by Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
