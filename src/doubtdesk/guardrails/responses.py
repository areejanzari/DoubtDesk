import random


OFF_TOPIC_RESPONSES = [
    "DoubtDesk is an academic tutor. Please ask a question related to your studies.",
    "I can help with academic doubts in subjects such as biology, chemistry, physics, and mathematics.",
    "This question is outside DoubtDesk's current academic scope. Try asking a study-related question.",
    "DoubtDesk is designed to explain academic concepts and help you understand your coursework.",
]


def get_off_topic_response() -> str:
    return random.choice(OFF_TOPIC_RESPONSES)