# 🎓 Agentic AI Study Coach & Learning Analytics Platform

> An autonomous, AI-driven educational platform that leverages classical machine learning for predictive analytics and a LangGraph-based agentic workflow for personalized, real-time study coaching.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_AI-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Predictive_ML-yellow)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

**🌐 Live Demo:** [https://studycoachgenai.streamlit.app/](https://studycoachgenai.streamlit.app/)

## 📖 Overview

This project is a full-stack AI application designed to support students through data-driven insights and interactive AI coaching. It combines **two core milestones** into a single, cohesive platform:

1. **Learning Analytics Engine (Predictive ML):** Analyzes historical student data (quiz scores, time spent, assignments) using Scikit-Learn to cluster learners into profiles (e.g., At-Risk, High-Performer) and predict their likelihood of passing.
2. **Agentic AI Study Coach (LLM + RAG):** An autonomous AI agent built with **LangGraph**. It ingests the student's ML-generated profile and dynamically adapts its coaching style. It uses Retrieval-Augmented Generation (RAG) via ChromaDB to proactively fetch and recommend targeted educational materials based on the student's conversational context and identified knowledge gaps.

## ✨ Key Features

- **Predictive Analytics Dashboard:** Interactive data visualizations (Seaborn/Matplotlib) showing student clusters (K-Means) and pass probability predictions (Logistic Regression).
- **Autonomous AI Agent:** Multi-step reasoning workflow orchestrated by LangGraph, enabling the AI to act as a stateful, goal-oriented coach.
- **RAG-Powered Tool Usage:** The agent autonomously queries a ChromaDB vector store containing educational content when it detects a student struggling with specific concepts.
- **Session Memory:** The agent maintains conversational context, allowing for fluid, continuous tutoring sessions.
- **Dynamic Personalization:** The system injects the ML prediction results directly into the agent's system prompt, ensuring the AI adjusts its tone, pacing, and difficulty level automatically.

## 🛠️ Technology Stack

- **Framework:** Streamlit (Frontend & Dashboard)
- **Agentic Orchestration:** LangGraph, LangChain Core
- **Machine Learning:** Scikit-Learn (K-Means, Logistic Regression), Pandas, NumPy
- **Vector Database (RAG):** ChromaDB, HuggingFace Embeddings (`all-MiniLM-L6-v2`)
- **Large Language Model:** Groq API (`llama-3.3-70b-versatile`) for ultra-fast, free-tier inference.

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/StudyCoach.git
cd StudyCoach
```

### 2. Create a Virtual Environment (Recommended)
It's best practice to use a virtual environment to manage dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\\Scripts\\activate`
```

### 3. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables
Copy the example environment file and add your Groq API key (get one for free at [groq.com](https://groq.com/)).
```bash
cp .env.example .env
# Edit .env and insert your GROQ_API_KEY
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 🧠 How It Works (Architecture)

1. **User interacts with the Dashboard** to generate/load student data.
2. **`ml_engine.py`** scales the data, clusters it via K-Means to identify the learner's "Profile", and runs a Logistic Regression model to predict their "Pass Probability".
3. **The User simulates a student** (adjusting sliders for scores/time), passing this new data point through the trained models.
4. **`agent.py`** initializes a LangGraph StateGraph. The ML profile is injected into the context.
5. **The User chats with the Coach**. The LLM uses Chain-of-Thought to determine if it should invoke the `search_educational_content` tool (RAG).
6. **`rag_setup.py`** retrieves relevant vector chunks from ChromaDB and returns them to the agent, which synthesizes a final personalized response.

## 🌐 Deployment

**The live application is deployed here:** [https://studycoachgenai.streamlit.app/](https://studycoachgenai.streamlit.app/)

This project is built to be easily deployed to platforms like **Streamlit Community Cloud** or **Hugging Face Spaces**.

### Streamlit Community Cloud
1. Push your repository to GitHub (ensure your `.env` and `chroma_db` are in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and connect your GitHub account.
3. Deploy the repository by selecting `app.py` as the main file.
4. In the Streamlit dashboard, go to **Advanced Settings -> Secrets** and add your API key: `GROQ_API_KEY="your_key"`.

### Hugging Face Spaces
1. Create a new Space on Hugging Face and select **Streamlit** as the SDK.
2. Push your files to the Space repository.
3. Go to the Space **Settings -> Variables and secrets** and add a new Secret: `GROQ_API_KEY`.

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/StudyCoach/issues).

## 📝 License
This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.
