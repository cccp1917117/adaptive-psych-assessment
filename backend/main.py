from fastapi import FastAPI
from pydantic import BaseModel
import json

from adaptive_engine import (
    select_initial_questions,
    score_answers,
    select_followup_questions
)

app = FastAPI()

with open("questions.json", "r", encoding="utf-8") as f:
    QUESTION_BANK = json.load(f)

class Answer(BaseModel):
    question_id: str
    score: int

class AnswerSet(BaseModel):
    answers: list[Answer]

@app.get("/")
def root():
    return {"message": "Adaptive Psych Assessment API Running"}

@app.get("/start")

def start_assessment():
    questions = select_initial_questions(QUESTION_BANK)

    return {
        "stage": "initial",
        "questions": questions
    }

@app.post("/next")

def next_questions(answer_set: AnswerSet):

    answers = [a.dict() for a in answer_set.answers]

    scores = score_answers(answers, QUESTION_BANK)

    answered_ids = [a["question_id"] for a in answers]

    next_qs = select_followup_questions(
        scores,
        QUESTION_BANK,
        answered_ids
    )

    return {
        "scores": scores,
        "next_questions": next_qs
    }