import random


OFF_TOPIC_RESPONSES = {
    "default": "DoubtDesk is an academic tutor. Please ask a question related to your studies.",
    "friendly": "I can help with academic doubts in subjects such as biology, chemistry, physics, and mathematics.",
    "direct": "This question is outside DoubtDesk's current academic scope. Try asking a study-related question.",
    "helpful": "DoubtDesk is designed to explain academic concepts and help you understand your coursework.",
}


def get_off_topic_response() -> str:
    return random.choice(list(OFF_TOPIC_RESPONSES.values()))