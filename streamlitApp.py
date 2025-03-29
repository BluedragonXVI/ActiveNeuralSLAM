import streamlit as st
from dqn_agent import DQNAgent
from drone_environment import DroneEnvironment

st.title("Drone DQN App")

env = DroneEnvironment()
agent = DQNAgent(env.state_dim, env.action_dim)

st.write("Drone DQN App is running...")
