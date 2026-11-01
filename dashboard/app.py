"""
Streamlit Dashboard for DRL Pothole Repair System
Real-time monitoring and visualization
"""

import streamlit as st
import sys
import os
sys.path.append('../data')
sys.path.append('../models')
sys.path.append('../environment')

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from repair_env import PotholeRepairEnv
from drl_agent import PotholeRepairAgent
import config

# Page config
st.set_page_config(
    page_title="DRL Pothole Repair System",
    page_icon="🛣️",
    layout="wide"
)

# Title
st.title("🛣️ DRL-Based Pothole Repair Scheduling System")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Model selection
model_path = st.sidebar.selectbox(
    "Select Model",
    ["./models/best_model", "./models/improved_model", "./models/baseline"]
)

# Simulation settings
num_episodes = st.sidebar.slider("Number of Episodes", 1, 20, 5)
timesteps_per_episode = st.sidebar.slider("Days per Episode", 30, 365, 365)

# Budget settings
initial_budget = st.sidebar.number_input(
    "Initial Budget ($)", 
    min_value=100000, 
    max_value=2000000, 
    value=500000,
    step=50000
)

# Run simulation button
run_simulation = st.sidebar.button("🚀 Run Simulation", type="primary")

# Cache environment creation
@st.cache_resource
def create_environment(budget):
    env = PotholeRepairEnv(
        num_segments=config.ENV_CONFIG['num_road_segments'],
        episode_length=config.ENV_CONFIG['episode_length'],
        initial_budget=budget
    )
    return env

@st.cache_resource
def load_agent(_env, model_path):
    agent = PotholeRepairAgent(env=_env, verbose=0)
    try:
        agent.load(model_path)
        return agent, True
    except:
        return agent, False

# Main content
if run_simulation:
    st.markdown("### 📊 Simulation Running...")
    
    # Create environment
    env = create_environment(initial_budget)
    agent, model_loaded = load_agent(env, model_path)
    
    if not model_loaded:
        st.warning("⚠️ Model not found! Using random policy.")
    else:
        st.success(f"✅ Model loaded: {model_path}")
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Metrics containers
    col1, col2, col3, col4 = st.columns(4)
    
    # Storage for results
    all_episodes_data = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        
        episode_data = {
            'safety_scores': [],
            'potholes': [],
            'budget': [],
            'repairs': [],
            'rewards': [],
            'actions': {'do_nothing': 0, 'patch': 0, 'overlay': 0, 'mill_fill': 0}
        }
        
        while not done and step < timesteps_per_episode:
            # Predict action
            if model_loaded:
                action, _ = agent.model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            # Step
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Record data
            episode_data['safety_scores'].append(info['network_safety'])
            episode_data['potholes'].append(info['total_potholes'])
            episode_data['budget'].append(info['budget_remaining'])
            episode_data['rewards'].append(reward)
            
            # Track actions
            action_types = ['do_nothing', 'patch', 'overlay', 'mill_fill']
            episode_data['actions'][action_types[action[1]]] += 1
            
            if 'repair_executed' in info and info['repair_executed']:
                episode_data['repairs'].append(step)
            
            step += 1
            done = terminated or truncated
            
            # Update progress
            progress = ((episode * timesteps_per_episode + step) / 
                       (num_episodes * timesteps_per_episode))
            progress_bar.progress(progress)
            status_text.text(f"Episode {episode+1}/{num_episodes}, Day {step}/{timesteps_per_episode}")
        
        all_episodes_data.append(episode_data)
    
    progress_bar.empty()
    status_text.empty()
    
    st.markdown("---")
    st.markdown("### 📈 Results")
    
    # Aggregate metrics
    final_safety_scores = [ep['safety_scores'][-1] for ep in all_episodes_data]
    final_potholes = [ep['potholes'][-1] for ep in all_episodes_data]
    total_repairs = [len(ep['repairs']) for ep in all_episodes_data]
    final_budgets = [ep['budget'][-1] for ep in all_episodes_data]
    avg_rewards = [np.mean(ep['rewards']) for ep in all_episodes_data]
    
    # Display summary metrics
    col1.metric("Avg Final Safety", f"{np.mean(final_safety_scores):.1f}", 
                delta=f"{np.mean(final_safety_scores) - 50:.1f}")
    col2.metric("Avg Repairs/Episode", f"{np.mean(total_repairs):.0f}")
    col3.metric("Avg Final Potholes", f"{np.mean(final_potholes):.0f}")
    col4.metric("Avg Budget Used", f"${initial_budget - np.mean(final_budgets):,.0f}")
    
    # Create visualizations
    st.markdown("#### 📊 Performance Over Time")
    
    # Plotly figure with subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Safety Score', 'Total Potholes', 'Budget Remaining', 'Cumulative Reward'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Plot each episode
    for i, ep_data in enumerate(all_episodes_data):
        days = list(range(len(ep_data['safety_scores'])))
        
        # Safety score
        fig.add_trace(
            go.Scatter(x=days, y=ep_data['safety_scores'], 
                      name=f'Episode {i+1}', 
                      mode='lines',
                      line=dict(width=1.5),
                      showlegend=True),
            row=1, col=1
        )
        
        # Potholes
        fig.add_trace(
            go.Scatter(x=days, y=ep_data['potholes'], 
                      name=f'Episode {i+1}',
                      mode='lines',
                      line=dict(width=1.5),
                      showlegend=False),
            row=1, col=2
        )
        
        # Budget
        fig.add_trace(
            go.Scatter(x=days, y=ep_data['budget'], 
                      name=f'Episode {i+1}',
                      mode='lines',
                      line=dict(width=1.5),
                      showlegend=False),
            row=2, col=1
        )
        
        # Cumulative reward
        cumulative_rewards = np.cumsum(ep_data['rewards'])
        fig.add_trace(
            go.Scatter(x=days, y=cumulative_rewards, 
                      name=f'Episode {i+1}',
                      mode='lines',
                      line=dict(width=1.5),
                      showlegend=False),
            row=2, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Day", row=2, col=1)
    fig.update_xaxes(title_text="Day", row=2, col=2)
    fig.update_yaxes(title_text="Safety Score", row=1, col=1)
    fig.update_yaxes(title_text="Potholes", row=1, col=2)
    fig.update_yaxes(title_text="Budget ($)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Reward", row=2, col=2)
    
    fig.update_layout(height=700, showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Action distribution
    st.markdown("#### 🔧 Action Distribution")
    
    # Aggregate actions across episodes
    total_actions = {'do_nothing': 0, 'patch': 0, 'overlay': 0, 'mill_fill': 0}
    for ep_data in all_episodes_data:
        for action_type, count in ep_data['actions'].items():
            total_actions[action_type] += count
    
    action_df = pd.DataFrame([
        {'Action': action, 'Count': count} 
        for action, count in total_actions.items()
    ])
    
    fig_actions = px.bar(
        action_df, 
        x='Action', 
        y='Count',
        title='Total Actions Taken Across All Episodes',
        color='Action',
        color_discrete_map={
            'do_nothing': '#FF6B6B',
            'patch': '#4ECDC4',
            'overlay': '#45B7D1',
            'mill_fill': '#96CEB4'
        }
    )
    
    st.plotly_chart(fig_actions, use_container_width=True)
    
    # Episode comparison table
    st.markdown("#### 📋 Episode Comparison")
    
    comparison_df = pd.DataFrame({
        'Episode': range(1, num_episodes + 1),
        'Final Safety': [f"{score:.1f}" for score in final_safety_scores],
        'Total Repairs': total_repairs,
        'Final Potholes': final_potholes,
        'Budget Used ($)': [f"{initial_budget - budget:,.0f}" for budget in final_budgets],
        'Avg Reward': [f"{reward:.2f}" for reward in avg_rewards]
    })
    
    st.dataframe(comparison_df, use_container_width=True)
    
    # Download results
    csv = comparison_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Results CSV",
        data=csv,
        file_name="drl_simulation_results.csv",
        mime="text/csv"
    )

else:
    # Welcome screen
    st.markdown("""
    ## 👋 Welcome to the DRL Pothole Repair System
    
    This dashboard allows you to:
    - 🎯 Test different trained models
    - 📊 Visualize agent performance over time
    - 🔧 Adjust simulation parameters
    - 📈 Compare multiple episodes
    
    ### 🚀 Getting Started
    1. Configure simulation parameters in the sidebar
    2. Select a trained model
    3. Click "Run Simulation" to see results
    
    ### 📌 Key Metrics
    - **Safety Score**: Network-wide road safety (0-100)
    - **Repairs**: Number of repair actions taken
    - **Budget**: Remaining budget after simulation
    - **Potholes**: Total active potholes at end
    """)
    
    # Show example metrics
    st.markdown("---")
    st.markdown("### 📊 System Overview")
    
    col1, col2, col3 = st.columns(3)
    col1.info(f"**Road Segments**: {config.ENV_CONFIG['num_road_segments']}")
    col2.info(f"**Episode Length**: {config.ENV_CONFIG['episode_length']} days")
    col3.info(f"**Default Budget**: ${config.ENV_CONFIG['initial_budget']:,}")
