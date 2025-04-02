"""
Federated Learning Dashboard

A Streamlit dashboard for monitoring the federated learning platform,
including model performance, contributions, and token economics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import aiohttp
import datetime
from typing import Dict, Any, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="Federated Learning Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Global variables
API_URL = "http://api:8000"  # Default to Docker service name
AUTH_TOKEN = None

# Helper functions
async def fetch_data(session, endpoint, headers=None):
    """Fetch data from API endpoint"""
    url = f"{API_URL}/api/{endpoint}"
    try:
        # Explicitly disable SSL verification for Docker internal communication
        async with session.get(url, headers=headers, ssl=False, timeout=5) as response:
            if response.status == 200:
                return await response.json()
            else:
                # Don't show error for every failed endpoint - we'll use mock data
                st.error(f"Error fetching {endpoint}: Status {response.status}")
                return None
    except Exception as e:
        # Don't show error for connection issues - we'll use mock data
        st.error(f"Connection error: {str(e)}")
        return None

async def fetch_all_data():
    """Fetch all required data from API endpoints"""
    headers = {"Authorization": f"Bearer {AUTH_TOKEN}"} if AUTH_TOKEN else None
    
    async with aiohttp.ClientSession() as session:
        # Always fetch stats
        stats_task = fetch_data(session, "stats")
        
        # Only fetch token balance if we have an auth token
        token_balance_task = fetch_data(session, "token/balance", headers) if AUTH_TOKEN else None
        
        # Mock data tasks
        mock_tasks = [
            mock_fetch_contributions(),
            mock_fetch_model_performance(),
            mock_fetch_governance_proposals()
        ]
        
        # Execute tasks
        stats = await stats_task
        token_balance = await token_balance_task if token_balance_task else None
        contributions, model_performance, governance = await asyncio.gather(*mock_tasks)
        
        return {
            "stats": stats,
            "token_balance": token_balance,
            "contributions": contributions,
            "model_performance": model_performance,
            "governance": governance
        }

async def mock_fetch_contributions():
    """Mock data for contributions"""
    # In a real implementation, this would fetch from the API
    contributions = []
    current_time = time.time()
    
    # Generate some mock contribution data
    for i in range(50):
        timestamp = current_time - (i * 3600)  # One contribution per hour
        contributions.append({
            "contribution_id": f"contrib_{i}",
            "client_id": f"client_{i % 10}",
            "timestamp": timestamp,
            "round": i // 5,
            "accuracy": 0.7 + (np.random.random() * 0.2),
            "loss": 0.5 - (np.random.random() * 0.3),
            "dataset_size": np.random.randint(100, 2000),
            "reward": np.random.randint(5, 50) * (10**18)
        })
    
    return contributions

async def mock_fetch_model_performance():
    """Mock data for model performance"""
    # In a real implementation, this would fetch from the API
    performance = []
    current_time = time.time()
    
    # Generate some mock model performance data
    for i in range(20):
        timestamp = current_time - (i * 86400)  # One data point per day
        performance.append({
            "timestamp": timestamp,
            "round": 20 - i,
            "accuracy": 0.7 + (i * 0.01),
            "loss": 0.5 - (i * 0.02),
            "participants": np.random.randint(5, 20)
        })
    
    return performance

async def mock_fetch_governance_proposals():
    """Mock data for governance proposals"""
    # In a real implementation, this would fetch from the API
    proposals = []
    current_time = time.time()
    
    # Generate some mock governance proposal data
    for i in range(5):
        timestamp = current_time - (i * 604800)  # One proposal per week
        voting_ends_at = timestamp + 604800  # 7 days voting period
        
        status = "Active"
        if i > 0:
            status = "Passed" if np.random.random() > 0.3 else "Rejected"
            if status == "Passed" and i > 1:
                status = "Executed"
        
        # Generate a random ethereum-like address
        addr = "".join([np.random.choice(list("0123456789abcdef")) for _ in range(40)])
        
        proposals.append({
            "proposal_id": i + 1,
            "title": f"Proposal {i + 1}",
            "description": f"Description for proposal {i + 1}",
            "proposer": f"0x{addr}",
            "created_at": timestamp,
            "voting_ends_at": voting_ends_at,
            "status": status,
            "yes_votes": np.random.randint(10000, 50000) * (10**18),
            "no_votes": np.random.randint(5000, 30000) * (10**18)
        })
    
    return proposals

def format_token_amount(amount_str):
    """Format token amount from wei to whole tokens"""
    try:
        amount = int(amount_str)
        return f"{amount / (10**18):,.2f}"
    except:
        return "0"

# UI functions
def render_sidebar():
    """Render sidebar with configuration options"""
    st.sidebar.title("Federated Learning Dashboard")
    
    global API_URL, AUTH_TOKEN
    
    API_URL = st.sidebar.text_input("API URL", value="http://api:8000")
    
    st.sidebar.subheader("Authentication")
    AUTH_TOKEN = st.sidebar.text_input("Authentication Token", type="password")
    
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("Dashboard Sections")
    show_overview = st.sidebar.checkbox("Platform Overview", value=True)
    show_model = st.sidebar.checkbox("Model Performance", value=True)
    show_contributions = st.sidebar.checkbox("Contributions", value=True)
    show_tokens = st.sidebar.checkbox("Token Economics", value=True)
    show_governance = st.sidebar.checkbox("Governance", value=True)
    
    st.sidebar.markdown("---")
    
    refresh_interval = st.sidebar.slider(
        "Auto Refresh (seconds)",
        min_value=0,
        max_value=300,
        value=60,
        step=10
    )
    
    if refresh_interval > 0:
        st.sidebar.info(f"Dashboard will refresh every {refresh_interval} seconds")
    
    return {
        "show_overview": show_overview,
        "show_model": show_model,
        "show_contributions": show_contributions,
        "show_tokens": show_tokens,
        "show_governance": show_governance,
        "refresh_interval": refresh_interval
    }

def render_overview(data):
    """Render platform overview section"""
    st.header("Platform Overview")
    
    stats = data.get("stats", {})
    if not stats:
        st.warning("No platform statistics available")
        return
    
    # Create top metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Contributions", f"{stats.get('total_contributions', 0):,}")
    
    with col2:
        st.metric("Active Clients", f"{stats.get('active_clients', 0):,}")
    
    with col3:
        st.metric("Current Round", f"{stats.get('current_round', 0):,}")
    
    with col4:
        st.metric("Model Version", stats.get('model_version', 'N/A'))
    
    # Create activity chart
    st.subheader("Platform Activity")
    
    # Process contributions for activity data
    contributions = data.get("contributions", [])
    if contributions:
        # Group contributions by day
        df = pd.DataFrame(contributions)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        daily_counts = df.groupby('date').size().reset_index(name='count')
        daily_counts['date'] = pd.to_datetime(daily_counts['date'])
        
        # Create chart
        fig = px.bar(
            daily_counts,
            x='date',
            y='count',
            title="Daily Contributions",
            labels={"date": "Date", "count": "Contributions"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No contribution data available for activity chart")

def render_model_performance(data):
    """Render model performance section"""
    st.header("Model Performance")
    
    performance = data.get("model_performance", [])
    if not performance:
        st.warning("No model performance data available")
        return
    
    # Process performance data
    df = pd.DataFrame(performance)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
    df = df.sort_values('timestamp')
    
    # Create metrics chart
    st.subheader("Performance Metrics Over Time")
    
    # Create subplot with shared x-axis
    fig = make_subplots(
        rows=2, 
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Accuracy", "Loss"),
        vertical_spacing=0.1
    )
    
    # Add accuracy trace
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['accuracy'],
            mode='lines+markers',
            name='Accuracy',
            marker=dict(color='green'),
            line=dict(width=2)
        ),
        row=1, col=1
    )
    
    # Add loss trace
    fig.add_trace(
        go.Scatter(
            x=df['date'],
            y=df['loss'],
            mode='lines+markers',
            name='Loss',
            marker=dict(color='red'),
            line=dict(width=2)
        ),
        row=2, col=1
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Participants per round
    st.subheader("Participants Per Round")
    
    fig = px.bar(
        df,
        x='round',
        y='participants',
        title="Number of Participants by Round",
        labels={"round": "Round", "participants": "Participants"}
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_contributions(data):
    """Render contributions section"""
    st.header("Contributions")
    
    contributions = data.get("contributions", [])
    if not contributions:
        st.warning("No contribution data available")
        return
    
    # Process contributions data
    df = pd.DataFrame(contributions)
    df['date'] = pd.to_datetime(df['timestamp'], unit='s')
    df['reward_tokens'] = df['reward'] / (10**18)
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Contributions", f"{len(df):,}")
    
    with col2:
        st.metric("Unique Clients", f"{df['client_id'].nunique():,}")
    
    with col3:
        st.metric("Average Accuracy", f"{df['accuracy'].mean():.2%}")
    
    # Create contribution quality chart
    st.subheader("Contribution Quality Distribution")
    
    fig = px.histogram(
        df,
        x='accuracy',
        nbins=20,
        title="Distribution of Contribution Accuracy",
        labels={"accuracy": "Accuracy", "count": "Contributions"}
    )
    fig.update_layout(bargap=0.1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Create reward correlation chart
    st.subheader("Reward Correlation")
    
    fig = px.scatter(
        df,
        x='accuracy',
        y='reward_tokens',
        color='dataset_size',
        size='dataset_size',
        hover_data=['client_id', 'date'],
        title="Reward vs Accuracy & Dataset Size",
        labels={
            "accuracy": "Accuracy",
            "reward_tokens": "Reward (Tokens)",
            "dataset_size": "Dataset Size"
        }
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Recent contributions table
    st.subheader("Recent Contributions")
    
    recent_df = df.sort_values('timestamp', ascending=False).head(10)
    recent_df['formatted_date'] = recent_df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')
    recent_df['formatted_reward'] = recent_df['reward_tokens'].apply(lambda x: f"{x:.2f}")
    
    st.dataframe(
        recent_df[['client_id', 'formatted_date', 'accuracy', 'dataset_size', 'formatted_reward']].rename(
            columns={
                'client_id': 'Client ID',
                'formatted_date': 'Date',
                'accuracy': 'Accuracy',
                'dataset_size': 'Dataset Size',
                'formatted_reward': 'Reward (Tokens)'
            }
        ),
        use_container_width=True
    )

def render_token_economics(data):
    """Render token economics section"""
    st.header("Token Economics")
    
    token_balance = data.get("token_balance", {})
    stats = data.get("stats", {})
    contributions = data.get("contributions", [])
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if token_balance:
            st.metric("Your Token Balance", format_token_amount(token_balance.get('balance', '0')))
        else:
            st.metric("Your Token Balance", "Login to view")
            st.info("Enter your authentication token in the sidebar to view your token balance")
    
    with col2:
        # Handle the case when stats is None
        rewards_issued = '0'
        if stats is not None:
            rewards_issued = stats.get('total_rewards_issued', '0')
        st.metric("Total Rewards Issued", format_token_amount(rewards_issued))
    
    with col3:
        st.metric("Platform Token Symbol", "FLT")
    
    # Create rewards over time chart
    st.subheader("Rewards Distribution Over Time")
    
    if contributions:
        # Process contributions for rewards data
        df = pd.DataFrame(contributions)
        df['date'] = pd.to_datetime(df['timestamp'], unit='s').dt.date
        df['reward_tokens'] = df['reward'] / (10**18)
        
        daily_rewards = df.groupby('date')['reward_tokens'].sum().reset_index()
        daily_rewards['date'] = pd.to_datetime(daily_rewards['date'])
        
        # Create chart
        fig = px.bar(
            daily_rewards,
            x='date',
            y='reward_tokens',
            title="Daily Rewards Distributed",
            labels={"date": "Date", "reward_tokens": "Tokens Distributed"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No reward data available for chart")
    
    # Create reward distribution by client chart
    st.subheader("Reward Distribution by Client")
    
    if contributions:
        # Process contributions for client rewards data
        client_rewards = df.groupby('client_id')['reward_tokens'].sum().reset_index()
        client_rewards = client_rewards.sort_values('reward_tokens', ascending=False).head(10)
        
        # Create chart
        fig = px.bar(
            client_rewards,
            x='client_id',
            y='reward_tokens',
            title="Top 10 Clients by Rewards",
            labels={"client_id": "Client ID", "reward_tokens": "Total Tokens Earned"}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No client reward data available for chart")

def render_governance(data):
    """Render governance section"""
    st.header("Governance")
    
    proposals = data.get("governance", [])
    if not proposals:
        st.warning("No governance data available")
        return
    
    # Process proposals data
    df = pd.DataFrame(proposals)
    df['created_date'] = pd.to_datetime(df['created_at'], unit='s')
    df['voting_end_date'] = pd.to_datetime(df['voting_ends_at'], unit='s')
    df['yes_tokens'] = df['yes_votes'] / (10**18)
    df['no_tokens'] = df['no_votes'] / (10**18)
    df['total_votes'] = df['yes_tokens'] + df['no_tokens']
    df['yes_percentage'] = (df['yes_tokens'] / df['total_votes'] * 100).fillna(0)
    
    # Create metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Proposals", f"{len(df):,}")
    
    with col2:
        active_count = len(df[df['status'] == 'Active'])
        st.metric("Active Proposals", f"{active_count:,}")
    
    with col3:
        executed_count = len(df[df['status'] == 'Executed'])
        st.metric("Executed Proposals", f"{executed_count:,}")
    
    # Create proposals list
    st.subheader("Proposals")
    
    # Display each proposal in an expandable section
    for _, row in df.iterrows():
        with st.expander(f"#{row['proposal_id']} - {row['title']} ({row['status']})"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**Description:** {row['description']}")
                st.markdown(f"**Proposer:** {row['proposer']}")
                st.markdown(f"**Created:** {row['created_date'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                if row['status'] == 'Active':
                    time_left = row['voting_end_date'] - datetime.datetime.now()
                    days_left = max(0, time_left.days)
                    st.markdown(f"**Voting Ends:** {row['voting_end_date'].strftime('%Y-%m-%d %H:%M:%S')} ({days_left} days left)")
            
            with col2:
                # Create a donut chart for vote distribution
                fig = go.Figure(go.Pie(
                    values=[row['yes_tokens'], row['no_tokens']],
                    labels=['Yes', 'No'],
                    hole=.4,
                    marker_colors=['#00CC96', '#EF553B']
                ))
                
                fig.update_layout(
                    title_text=f"Votes: {int(row['yes_percentage'])}% Yes",
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(t=30, b=10, l=10, r=10),
                    height=200
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add vote buttons (these would need to be connected to the API in a real implementation)
                if row['status'] == 'Active':
                    col1, col2 = st.columns(2)
                    with col1:
                        st.button("Vote Yes", key=f"yes_{row['proposal_id']}")
                    with col2:
                        st.button("Vote No", key=f"no_{row['proposal_id']}")

# Main function
def main():
    """Main dashboard function"""
    # Title
    st.title("Decentralized Federated Learning Dashboard")
    
    # Render sidebar and get settings
    settings = render_sidebar()
    
    # Fetch all data
    data = asyncio.run(fetch_all_data())
    
    # Set up tabs
    if settings["show_overview"]:
        render_overview(data)
        st.markdown("---")
    
    if settings["show_model"]:
        render_model_performance(data)
        st.markdown("---")
    
    if settings["show_contributions"]:
        render_contributions(data)
        st.markdown("---")
    
    if settings["show_tokens"]:
        render_token_economics(data)
        st.markdown("---")
    
    if settings["show_governance"]:
        render_governance(data)
    
    # Auto-refresh logic
    if settings["refresh_interval"] > 0:
        time.sleep(1)  # Small delay to prevent excessive resource usage
        st.experimental_rerun()

if __name__ == "__main__":
    main()
