# Player Recommendation System

This is a Machine Learning-based Player Recommendation System built using Streamlit. It allows users to upload player datasets, train models, and get intelligent player recommendations.

 Features

- Upload player attribute data (CSV format)
- Handle missing values automatically
- Choose from various ML models (KMeans, DBSCAN, Decision Tree, Random Forest, etc.)
- Train and visualize model performance
- Input player attributes manually to get prediction: Recommended or Not Recommended
- View cluster plots for unsupervised learning

Live Application

Access the deployed application here:  
https://player-recommendation-system-j4vdv6qdv4foa7zxc4avas.streamlit.app/

 Getting Started (Local Setup)

 1. Clone the Repository

```bash
git clone https://github.com/Balu1724/player-recommendation-system.git
cd player-recommendation-system

2. Create a Virtual Environment (Optional but Recommended)

python -m venv venv
# For Windows
venv\Scripts\activate
# For Mac/Linux
source venv/bin/activate

3.Install Dependencies

pip install -r requirements.txt

4. Run the Application

streamlit run app.py
