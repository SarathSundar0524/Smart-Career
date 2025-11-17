# Smart-Career
Personalized Learning Path Recommender: An AI-powered Streamlit app that recommends the most suitable online courses based on a user's skills, background, and career goals. Uses Sentence Transformer embeddings and cosine similarity to rank courses and generate a personalized learning path.
Personalized Learning Path Recommender

A personalized, AI-assisted learning recommender system that analyzes a user's background, skills, and goals to suggest the most suitable online courses along with a structured learning path.

This project uses Streamlit for the UI and Sentence Transformers for embedding-based similarity search.

Features

Embedding-based course matching using SentenceTransformer

Cosine similarity for relevance scoring

Prerequisite checking and penalty-based fit scoring

Short-term and long-term learning plan generation

Downloadable JSON output

Clean, interactive Streamlit interface

Project Structure
learning-path-recommender/
│── app.py                # Streamlit application
│── requirements.txt      # Dependencies
│── README.md             # Documentation
│── /data
│     └── courses.csv     # Course dataset
│── /screenshots
      └── ui.png          # Optional UI screenshot

Installation
1. Clone the repository
git clone https://github.com/your-username/learning-path-recommender.git
cd learning-path-recommender

2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

3. Install dependencies
pip install -r requirements.txt

Running the Application
streamlit run app.py


The app will open automatically at:

http://localhost:8501

How the System Works

Loads a curated course dataset

Prepares course text descriptions

Converts each course description into an embedding

Converts user profile into an embedding

Computes similarity between the user and each course

Applies prerequisite checks and level penalties

Calculates a final fit score

Outputs:

Top recommended courses

A structured learning timeline

Downloadable JSON output

Sample JSON Output
{
  "profile_id": "User",
  "recommendations": [
    {
      "title": "Python for Everybody",
      "fit_score": 87,
      "rationale": "Matches your background...",
      "gap": "Fills missing basic foundations...",
      "prep": "No preparation needed."
    }
  ],
  "timeline": {
    "short_term_sequence": "Python for Everybody (8w)",
    "long_term_sequence": "Machine Learning by Andrew Ng (11w)",
    "explanation": "Short-term courses are 12 weeks or less with a high fit score."
  }
}

Optional: Deploy on Streamlit Cloud

Push the repository to GitHub

Visit: https://share.streamlit.io

Connect your GitHub repository

Select app.py as the entry file

Deploy

Requirements File (requirements.txt)
streamlit
pandas
numpy
sentence-transformers
scikit-learn
openai
