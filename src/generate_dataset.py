"""
Dataset Generator for Kaggle-style Sentiment Analysis Dataset.
Produces both CSV and JSON formats covering multi-domain feedback:
- Product reviews (Amazon/E-commerce)
- Movies & Entertainment (IMDB/Rotten Tomatoes)
- Customer Service & Airlines (Twitter Sentiment)
- Tech & App reviews (Play Store/App Store)
- Short social media posts (Twitter/X, Reddit)
"""

import os
import json
import random
import pandas as pd

# Set deterministic seed
random.seed(42)

POSITIVE_TEMPLATES = [
    # General & Short
    "Wow! {adj_pos} {noun}.",
    "Absolutely {adj_pos}! I would definitely recommend this to everyone.",
    "This is {adv_pos} {adj_pos}, exceeded all my expectations!",
    "Hands down the best {noun} I have used in years.",
    "Remarkable quality and {adj_pos} performance. Five stars!",
    "So happy with my purchase! It works {adv_pos} well.",
    "Incredible experience! The {noun} was {adj_pos} and customer support was very helpful.",
    "I'm truly impressed with how {adj_pos} this is. Worth every penny.",
    "Brilliant work! {adv_pos} {adj_pos} and easy to use.",
    "Super satisfied! Fast delivery, great packaging, and {adj_pos} results.",
    "Wow, just wow! Outstanding quality and craftsmanship.",
    "A masterpiece! Truly {adj_pos} from beginning to end.",
    "Top notch! Highly reliable and delightfully {adj_pos}.",
    "I wasn't sure at first, but now I cannot live without this {noun}.",
    "Five out of five stars! Perfectly crafted and wonderfully {adj_pos}.",
    "Great value for money. Very {adj_pos} and robust.",
    "Loved everything about this! Will definitely buy again.",
    "The best decision I made this month. {adv_pos} {adj_pos}.",
    "Such a breath of fresh air! Clean, fast, and {adj_pos}.",
    "Fantastic customer support resolved my inquiry in minutes with {adj_pos} care.",
    # Negated negative phrases (challenging NLP cases that mean positive)
    "Honestly, not bad at all for the price! Really surprised me.",
    "I cannot complain, it works exactly as needed.",
    "No regrets whatsoever, turned out much better than expected.",
    "Was not disappointed with the performance, truly solid.",
    "Not terrible at all, actually quite {adj_pos} once you set it up.",
    "Zero complaints so far! Everything runs smoothly.",
    # Domain specific - Movies
    "The cinematography and acting were absolutely {adj_pos}.",
    "An inspiring story with {adj_pos} performances by the entire cast.",
    "A thrill from start to finish! The soundtrack was especially {adj_pos}.",
    # Domain specific - Tech & Apps
    "The new app update is {adv_pos} smooth and responsive.",
    "Zero bugs encountered, clean UI, and {adj_pos} design.",
    "Setup took less than two minutes. Works like a charm!",
    # Domain specific - Food & Restaurant
    "The food was delicious and the ambiance was {adj_pos}.",
    "Best meal we have had in a very long time! Highly recommended.",
    # Domain specific - Airline & Travel
    "Flight was on time, staff was courteous, and seats were {adj_pos}.",
    "Seamless check-in and friendly crew throughout the entire journey.",
    # Social media noise
    "OMG loved this so much!! 😍❤️ @brand #bestever",
    "Wow... just got mine in the mail and it's {adj_pos}! https://example.com/item"
]

NEGATIVE_TEMPLATES = [
    # General & Short
    "Terrible experience. The {noun} is completely {adj_neg}.",
    "Horrible quality! Do not waste your money on this {noun}.",
    "Extremely disappointed with this purchase. It broke after two days.",
    "Worst {noun} ever. It does not work as advertised.",
    "Awful customer service, nobody responded to my emails for two weeks.",
    "Very poor craftsmanship. Cheap material and {adj_neg} finish.",
    "I regret buying this! Completely useless and {adj_neg}.",
    "Waste of time and money. Totally {adj_neg}.",
    "Defective item received. Not worth the hassle.",
    "Frustrating! Constant crashes and {adj_neg} performance.",
    "Do not buy! The product stopped working almost immediately.",
    "Very bad quality, arrived broken and dirty. Refund demanded.",
    "I expected much better, but this was utterly {adj_neg}.",
    "Failed to deliver on promises. Extremely {adj_neg} results.",
    "Zero stars if I could. Absolutely {adj_neg} service.",
    "Not worth the price tag at all. Horribly {adj_neg}.",
    "Avoid at all costs! You will be deeply disappointed.",
    "Terrible build quality. Fell apart within an hour of light use.",
    "Rude staff and completely unprofessional attitude.",
    "Misleading product description. What arrived was totally {adj_neg}.",
    # Negated positive phrases (challenging NLP cases that mean negative)
    "Not good at all. Died within a single day.",
    "Definitely not worth the hype, very disappointing.",
    "Not happy with this purchase, expected way more.",
    "Was not impressed by the battery life or build.",
    "I would not recommend this {noun} to anyone.",
    "Not great, frankly quite {adj_neg}.",
    # Domain specific - Movies
    "Boring plot, wooden acting, and completely {adj_neg} pacing.",
    "Walked out halfway through. An uninspired, dull film.",
    "Waste of two hours. Poor dialogue and predictable story.",
    # Domain specific - Tech & Apps
    "The latest update completely ruined the app. Keeps freezing.",
    "Buggy software full of glitches and terrible navigation.",
    "Battery drains in an hour after installing. Uninstalling now.",
    # Domain specific - Food & Restaurant
    "Cold food, slow service, and {adj_neg} hygiene.",
    "We waited over an hour for our food and it was completely undercooked.",
    # Domain specific - Airline & Travel
    "Flight delayed by six hours without any explanation or voucher.",
    "Lost my baggage and refused to take any responsibility. Terrible!",
    # Social media noise
    "Never buying from @store again!! Horrible support 😡👎",
    "Total waste of money! http://badreviews.com/complaint"
]

NEUTRAL_TEMPLATES = [
    # General & Short
    "The package arrived on Tuesday as scheduled.",
    "The {noun} is average. It does what it says, nothing more, nothing less.",
    "Standard quality for this price range. Neither good nor bad.",
    "The item is okay, regular size and normal weight.",
    "Received the order today. Box was standard brown cardboard.",
    "It works as expected according to the specifications.",
    "Nothing special about this {noun}, just an ordinary option.",
    "The color is slightly darker than the picture, but acceptable.",
    "Average performance overall. Mediocre build but functional.",
    "It is functional, but lacks advanced features found in newer models.",
    "Shipped in standard packaging. Included the manual and power cable.",
    "The item measures approximately 10 inches and weighs 2 pounds.",
    "Purchased this for work. It serves its basic intended purpose.",
    "Standard delivery time of 4 business days.",
    "The dimensions match the description in the user guide.",
    "It comes with a standard 12-month manufacturer warranty.",
    "Neutral opinion so far. Will update after using it for a month.",
    "Battery life lasts around 6 hours during normal office use.",
    "The menu options are standard for this category of products.",
    "Replaced my previous unit with this one yesterday.",
    "It has pros and cons. Some parts are fine while others are average.",
    "Decent, not amazing but not terrible either.",
    "Just a routine purchase. Does what a basic {noun} should do.",
    # Domain specific - Movies
    "The movie runtime is 118 minutes and it is rated PG-13.",
    "An average thriller film with typical genre tropes and predictable twists.",
    "The documentary focuses on the historical timeline of the event.",
    # Domain specific - Tech & Apps
    "App version 2.4 released with updated privacy policy terms.",
    "The software requires 4GB of RAM and Windows 10 or later.",
    "Device charges via standard USB-C cable.",
    # Domain specific - Food & Restaurant
    "The restaurant is open from 11 AM to 10 PM on weekdays.",
    "A standard lunch menu with sandwiches and beverages.",
    # Domain specific - Airline & Travel
    "The flight departed at 14:30 and arrived at 17:15 local time.",
    "Carry-on baggage allowance is limited to 7 kilograms per passenger.",
    # Social media noise
    "Just received package from @store today #delivery #update",
    "Tracking info says delivered to front porch. http://track.pkg/123"
]

ADJECTIVES_POS = [
    "amazing", "fantastic", "excellent", "superb", "wonderful", "terrific",
    "fabulous", "outstanding", "impressive", "flawless", "exceptional",
    "delightful", "splendid", "top-tier", "brilliant", "great", "stellar"
]

ADVERBS_POS = [
    "remarkably", "exceptionally", "incredibly", "extremely", "truly",
    "wonderfully", "consistently", "delightfully", "surprisingly", "thoroughly"
]

ADJECTIVES_NEG = [
    "terrible", "horrible", "awful", "dreadful", "abysmal", "pathetic",
    "useless", "faulty", "garbage", "trash", "disastrous", "unacceptable",
    "atrocious", "flawed", "subpar", "defective", "worthless"
]

NOUNS = [
    "product", "device", "service", "item", "movie", "application",
    "gadget", "software", "delivery", "experience", "purchase", "meal",
    "laptop", "phone", "headphone", "platform", "tool", "hotel"
]

CATEGORIES = [
    "Product Review", "Movie Review", "Customer Support",
    "Tech & Software", "Travel & Airlines", "Food & Dining"
]


def generate_samples(count_per_class: int = 1200):
    samples = []
    sample_id = 10001

    # Generate Positive
    for _ in range(count_per_class):
        tmpl = random.choice(POSITIVE_TEMPLATES)
        text = tmpl.format(
            adj_pos=random.choice(ADJECTIVES_POS),
            adv_pos=random.choice(ADVERBS_POS),
            noun=random.choice(NOUNS)
        )
        cat = random.choice(CATEGORIES)
        samples.append({
            "id": sample_id,
            "text": text,
            "sentiment": "Positive",
            "category": cat,
            "source": "Kaggle Benchmark NLP Dataset"
        })
        sample_id += 1

    # Generate Negative
    for _ in range(count_per_class):
        tmpl = random.choice(NEGATIVE_TEMPLATES)
        text = tmpl.format(
            adj_neg=random.choice(ADJECTIVES_NEG),
            noun=random.choice(NOUNS)
        )
        cat = random.choice(CATEGORIES)
        samples.append({
            "id": sample_id,
            "text": text,
            "sentiment": "Negative",
            "category": cat,
            "source": "Kaggle Benchmark NLP Dataset"
        })
        sample_id += 1

    # Generate Neutral
    for _ in range(count_per_class):
        tmpl = random.choice(NEUTRAL_TEMPLATES)
        text = tmpl.format(
            noun=random.choice(NOUNS)
        )
        cat = random.choice(CATEGORIES)
        samples.append({
            "id": sample_id,
            "text": text,
            "sentiment": "Neutral",
            "category": cat,
            "source": "Kaggle Benchmark NLP Dataset"
        })
        sample_id += 1

    # Shuffle dataset
    random.shuffle(samples)
    return samples


def main():
    os.makedirs("data", exist_ok=True)
    samples = generate_samples(1200)  # Total 3600 samples
    df = pd.DataFrame(samples)

    csv_path = os.path.join("data", "sentiment_dataset.csv")
    json_path = os.path.join("data", "sentiment_dataset.json")

    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(df)} samples across 3 sentiment classes.")
    print(f"CSV saved to: {csv_path}")
    print(f"JSON saved to: {json_path}")
    print("\nClass distribution:")
    print(df["sentiment"].value_counts())


if __name__ == "__main__":
    main()
