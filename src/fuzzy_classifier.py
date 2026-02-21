from rapidfuzz import fuzz 
def classify_letter(text):
    candidates ={
        "chief_engineer": "chief engineer",
        "secretary": "secretary",
        "chair_person" : "chair person"
    }

    best_match = None
    highest_score = 0

    for category,phrase in candidates.items():
        score = fuzz.partial_ratio(text,phrase)

        if score > highest_score:
            highest_score = score
            best_match = category

        print("Matching score : ", highest_score)

        if highest_score > 70:
            return best_match
        else:
            return "uncertain"