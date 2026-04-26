import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ml_engine import MLEngine
from agent import run_agent

# Set page config
st.set_page_config(
    page_title="Agentic AI Study Coach",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI Aesthetics
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background: #ffffff;
    }
    h1 {
        color: #1E3A8A;
        font-family: 'Inter', sans-serif;
    }
    h2, h3 {
        color: #2563EB;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
    }
    .metric-card {
        background-color: white;
        color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .metric-card h3 {
        color: #1E3A8A;
    }
    .metric-card p, .metric-card b {
        color: #1E293B !important;
    }
    
    /* Floating Chat Input Bar */
    div[data-testid="stChatInput"] {
        padding-bottom: 30px !important;
        background-color: transparent !important;
    }
    div[data-testid="stChatInput"] > div {
        max-width: 800px !important;
        width: 100% !important;
        margin: 0 auto !important;
        border-radius: 25px !important;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4) !important;
        border: 1px solid #334155 !important;
        background-color: #1E293B !important;
    }
    
    /* Ensure the text area inside looks seamless */
    div[data-testid="stChatInputTextArea"] {
        border-radius: 25px !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'ml_engine' not in st.session_state:
    st.session_state.ml_engine = MLEngine()
    
if 'df' not in st.session_state:
    st.session_state.df = None
    
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'student_profile' not in st.session_state:
    st.session_state.student_profile = "Unknown"
    
if 'pass_prob' not in st.session_state:
    st.session_state.pass_prob = 0.0

st.title("🎓 Agentic AI Study Coach & Learning Analytics Platform")

# Layout
tab1, tab2 = st.tabs(["📊 Analytics Dashboard (M1)", "💬 AI Study Coach (M2)"])

with tab1:
    st.header("Learning Analytics & Predictive Modeling")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Data Setup")
        if st.button("Generate Synthetic Student Data & Train Model"):
            with st.spinner("Generating data and training ML models..."):
                df = st.session_state.ml_engine.generate_synthetic_data()
                trained_df = st.session_state.ml_engine.train(df)
                st.session_state.df = trained_df
            st.success(f"Model trained successfully! Accuracy: {st.session_state.ml_engine.accuracy:.2%}")
            
        st.subheader("Simulate a Student")
        if st.session_state.ml_engine.is_trained:
            quiz_score = st.slider("Average Quiz Score", 0, 100, 75)
            time_spent = st.slider("Hours Spent Studying", 0, 100, 20)
            assignments = st.slider("Assignments Completed", 0, 10, 5)
            
            if st.button("Analyze Student"):
                prediction = st.session_state.ml_engine.predict_student(quiz_score, time_spent, assignments)
                
                # Update session state for the Agent
                # Assuming simple mapping for MVP
                profile_map = {0: "Average", 1: "High-Performer", 2: "At-Risk"}
                profile = profile_map.get(prediction["cluster_id"], "Average")
                
                st.session_state.student_profile = profile
                st.session_state.pass_prob = prediction["pass_probability"]
                
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Analysis Results</h3>
                    <p><b>Learner Profile:</b> {profile}</p>
                    <p><b>Pass Probability:</b> {prediction["pass_probability"]:.2%}</p>
                    <p><b>Prediction:</b> {'Pass' if prediction['pass_prediction'] else 'Fail'}</p>
                </div>
                """, unsafe_allow_html=True)
                
    with col2:
        if st.session_state.df is not None:
            st.subheader("Dataset Overview")
            st.dataframe(st.session_state.df.head(5))
            
            st.subheader("Learner Clusters")
            fig, ax = plt.subplots()
            sns.scatterplot(
                data=st.session_state.df, 
                x='time_spent_hours', 
                y='quiz_score', 
                hue='learner_profile',
                palette='viridis',
                ax=ax
            )
            ax.set_title("Student Clustering by Performance and Effort")
            st.pyplot(fig)
        else:
            st.info("Click 'Generate Synthetic Student Data & Train Model' to view analytics.")

with tab2:
    st.header("Personalized Agentic AI Study Coach")
    
    if st.session_state.student_profile == "Unknown":
        st.warning("⚠️ Please go to the Analytics Dashboard, train the model, and 'Analyze a Student' to initialize your profile before chatting!")
    else:
        st.success(f"Coach Context Active — Profile: {st.session_state.student_profile} | Predicted Pass Rate: {st.session_state.pass_prob:.0%}")
        
        # Display chat history
        for msg in st.session_state.chat_history:
            if msg.type == "human":
                st.chat_message("user").write(msg.content)
            elif msg.type == "ai" and msg.content.strip():
                st.chat_message("assistant").write(msg.content)
                
        # Chat input
        if prompt := st.chat_input("Ask your AI Coach for help..."):
            st.chat_message("user").write(prompt)
            
            with st.spinner("Thinking..."):
                try:
                    messages = run_agent(
                        user_input=prompt,
                        student_profile=st.session_state.student_profile,
                        pass_probability=st.session_state.pass_prob,
                        memory=st.session_state.chat_history
                    )
                    
                    # Update history
                    st.session_state.chat_history = messages
                    
                    # The last message is the AI's response
                    ai_msg = [m for m in messages if m.type == "ai"][-1]
                    st.chat_message("assistant").write(ai_msg.content)
                except Exception as e:
                    st.error(f"Error communicating with AI: {e}\\n\\nPlease ensure you have set GROQ_API_KEY in the .env file.")
