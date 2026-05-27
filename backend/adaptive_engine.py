import random

def select_initial_questions(question_bank, n=3):
    return random.sample(question_bank, min(n, len(question_bank)))

def score_answers(answers, question_bank):
    scores = {}

    for ans in answers:
        qid = ans["question_id"]
        value = ans["score"]

        question = next(q for q in question_bank if q["id"] == qid)
        dim = question["dimension"]

        if dim not in scores:
            scores[dim] = []

        scores[dim].append(value)

    return {
        dim: sum(vals) / len(vals)
        for dim, vals in scores.items()
    }

def select_followup_questions(scores, question_bank, answered_ids, n=2):
    if not scores:
        return []

    sorted_dims = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    target_dim = sorted_dims[0][0]

    candidates = [
        q for q in question_bank
        if q["dimension"] == target_dim and q["id"] not in answered_ids
    ]

    return random.sample(candidates, min(n, len(candidates)))