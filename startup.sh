#!/bin/bash

# Install Python packages
pip install -r requirements.txt

# Start your application
# streamlit run streamlit_app_agent.py
streamlit run streamlit_app_agent.py --server.port=$PORT --server.address=0.0.0.0
