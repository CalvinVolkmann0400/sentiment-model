from transformers import pipeline

classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
)

labels = ["ready to buy now", "just exploring", "can't afford it"]
answers = [
    "I've been trying to scale my coaching business for a year and I'm ready to invest in help as soon as possible.",
    "I'm just curious what you offer, maybe I'll look into it sometime next year.",
    "I love the idea, but I really don't have the money for this right now.",
]

for answer in answers:
    print(classifier(answer, candidate_labels=labels))
