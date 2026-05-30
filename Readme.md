# Adaptive Psych Assessment

AI-driven adaptive psychological assessment system.

## Features
- Adaptive question selection
- Dynamic scoring
- FastAPI backend
- Extensible question bank

## Tech Stack
- FastAPI
- Python
- React (planned)

## Roadmap
- [x] Backend prototype
- [ ] Frontend UI
- [ ] Database
- [ ] IRT/CAT
- [ ] LLM adaptive interviewing

## Quick Start
Pre-requirement: Got streamlit and requests downloaded in the frontend directory by running the command:
```pip install streamlit requests```  

Step 1: Open a terminal and move to the backend directory to run the command:```uvicorn main:app --reload```  
Step 2: Open a second terminal and move to the frontend directory to run the command:```streamlit run app.py```  