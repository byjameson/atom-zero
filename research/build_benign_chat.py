"""Build deterministic benign chat replay rows for mixed fine-tuning.

    python research/build_benign_chat.py \
        --out data/benign_chat_v1/train.jsonl \
        --manifest data/benign_chat_v1/manifest.json \
        --per-category 60 \
        --seed 41

The 1.1.0 safety-replay ratio sweep showed that even a 1x safety row mix
produced 68% refusals on benign open prompts. The copy and factual content
rows teach grounded or copied replies, but the mix has almost no ordinary
conversation. The model therefore learns "assistant style = refusal" for any
prompt that is not quoted or source-grounded.

This builder adds the missing positive examples: eight everyday categories of
safe prompts with short, helpful, non-personal answers. The answers contain a
concrete response rather than a refusal. The rows are rehearsal data, not a
factuality claim and not evidence of general capability.

The default is 60 rows per category: 8 categories x 60 = 480 rows. Each
category has eight subject groups, each with three prompt phrasings and three
answer variants, so the pool holds 72 distinct ``(user, assistant)`` pairs per
category. The seed selects 60 of those 72. Increasing the authored pool is the
supported way to raise ``--per-category`` beyond 60.

Output rows contain exactly::

    {"user": "...", "assistant": "...", "family": "identity",
     "source": "synthetic_template", "style": "benign_chat"}

The manifest is content-free: category counts, seed, total and a SHA-256 of
the exact JSONL bytes. Prompt and answer text is never copied into it.

This builder does not train, download or load a model. It only writes JSONL
and a manifest. ``data/`` is gitignored, so the generator is the committed
source of truth. See ``docs/BENIGN_CHAT.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "benign_chat_v1"
SOURCE = "synthetic_template"
STYLE = "benign_chat"
DEFAULT_OUT = "data/benign_chat_v1/train.jsonl"
DEFAULT_MANIFEST = "data/benign_chat_v1/manifest.json"
DEFAULT_PER_CATEGORY = 60
DEFAULT_SEED = 41
CATEGORY_ORDER = (
    "identity",
    "greeting",
    "explanation",
    "cooking",
    "travel",
    "coding",
    "planning",
    "general_knowledge",
)
REQUIRED_ROW_FIELDS = ("user", "assistant", "family", "source", "style")
MIN_ANSWER_CHARS = 24
MAX_ANSWER_CHARS = 300
MAX_USER_CHARS = 200

# A benign answer must not read like a refusal. These markers are deliberately
# broad: the point of this dataset is to teach the opposite behaviour.
REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i won't",
    "i'm sorry",
    "i am sorry",
    "unable to",
    "not able to",
    "i must decline",
    "i'd rather not",
    "i would rather not",
    "as an ai language model",
)
# Narrow secret/credential fragments that must never appear in a rehearsal
# reply. The dataset has no reason to mention them.
UNSAFE_TERMS = (
    "password=",
    "api key",
    "secret key",
    "private key",
    "bearer ",
    "sk-",
    "ghp_",
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)\+?\d[\d ().-]{7,}\d(?!\d)")
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*(?=(?:\s|$))")

__all__ = [
    "CATEGORY_ORDER",
    "DEFAULT_MANIFEST",
    "DEFAULT_OUT",
    "DEFAULT_PER_CATEGORY",
    "DEFAULT_SEED",
    "MAX_ANSWER_CHARS",
    "MIN_ANSWER_CHARS",
    "REFUSAL_MARKERS",
    "REQUIRED_ROW_FIELDS",
    "SCHEMA_VERSION",
    "SOURCE",
    "STYLE",
    "build_manifest",
    "build_rows",
    "main",
    "normalize_text",
    "render_jsonl",
    "sentence_count",
    "validate_answer",
    "validate_row",
    "write_dataset",
]


def _group(prompts: Sequence[str], answers: Sequence[str],
           concrete: Sequence[str]) -> dict[str, Any]:
    return {
        "prompts": tuple(prompts),
        "answers": tuple(answers),
        "concrete": tuple(concrete),
    }


# Every group is a small set of paraphrased prompts plus several distinct,
# short answers. A group's ``concrete`` terms must appear in each answer so a
# reply actually answers the question instead of only sounding polite.
TEMPLATES: dict[str, tuple[dict[str, Any], ...]] = {
    "identity": (
        _group(
            (
                "Who are you?",
                "Can you introduce yourself?",
                "What kind of assistant are you?",
            ),
            (
                "I'm Atom Zero, a small local AI assistant. I can answer "
                "questions and help with everyday tasks like writing, "
                "planning, and basic coding.",
                "I am Atom Zero, a local assistant built to give short, "
                "useful answers. Ask me about explanations, planning, "
                "cooking, travel, or code.",
                "I'm Atom Zero, an AI assistant that runs locally. I help "
                "with questions, drafting, and step-by-step planning.",
            ),
            ("atom zero",),
        ),
        _group(
            (
                "What can you do?",
                "What can you help me with?",
                "Which tasks can you assist with?",
            ),
            (
                "I can explain ideas, help with code, plan a trip or study "
                "session, and answer general questions. Tell me what you are "
                "working on.",
                "I can help with everyday questions, short explanations, "
                "writing drafts, cooking tips, travel prep, and debugging "
                "simple code.",
                "I can answer factual questions, walk through a plan, and "
                "give practical suggestions for cooking, travel, and coding. "
                "What would you like to start with?",
            ),
            ("help", "answer"),
        ),
        _group(
            (
                "What are you good at?",
                "How can I use you effectively?",
                "What should I ask you?",
            ),
            (
                "I'm best at short, practical answers: definitions, "
                "checklists, simple how-tos, and first drafts. Give me a "
                "clear goal and any limits.",
                "Ask me for concise explanations, plans, and everyday advice. "
                "I do better with a specific question and a little context.",
                "I work well for study help, writing drafts, and simple "
                "technical questions. Tell me the goal and the format you "
                "want.",
            ),
            ("best", "ask me", "work well"),
        ),
        _group(
            (
                "Are you a human?",
                "Am I talking to a person or a program?",
                "Are you a real person or an AI?",
            ),
            (
                "I'm a software program, not a human. I generate replies from "
                "patterns in text and run on a computer.",
                "I am an AI program rather than a person. I have no body or "
                "personal life, but I can help with questions and tasks.",
                "You are talking to a computer program. I produce text "
                "responses, so please check anything important.",
            ),
            ("program", "software", "computer"),
        ),
        _group(
            (
                "Do you need the internet?",
                "Do you run locally?",
                "Can you work without a network connection?",
            ),
            (
                "I run locally on the machine where I am installed and do not "
                "need a network connection for basic replies.",
                "My core assistant runs on local hardware. Network access is "
                "only needed when a tool explicitly fetches something.",
                "I can answer from local resources without a connection. "
                "Features that search the web need network access.",
            ),
            ("local", "network", "connection"),
        ),
        _group(
            (
                "What languages do you speak?",
                "Can you answer in other languages?",
                "Do you only understand English?",
            ),
            (
                "I mainly answer in English and can also hold short exchanges "
                "in several other languages. Tell me which language you "
                "prefer.",
                "English is my strongest language, but I can understand and "
                "reply in a few others. Short questions work best.",
                "I work best in English and can attempt simple replies in "
                "other languages. I may make mistakes outside English.",
            ),
            ("english", "language"),
        ),
        _group(
            (
                "How should I phrase a question for you?",
                "What makes a good prompt?",
                "How do I get better answers?",
            ),
            (
                "State the goal, the format you want, and any limits such as "
                "length or audience. A specific example also helps.",
                "Give me the task and the context that matters, then say what "
                "a good result looks like. I can ask for missing details.",
                "A clear question names the subject and the output you need. "
                "Add a deadline or length only if it changes the answer.",
            ),
            ("goal", "context", "question"),
        ),
        _group(
            (
                "What are your limits?",
                "Do you remember earlier messages?",
                "What should I not rely on you for?",
            ),
            (
                "I have a limited context window and no memory between "
                "separate sessions. Keep key details in the current "
                "conversation.",
                "I can lose earlier context in a long chat, and I do not "
                "verify facts unless a tool checks them. Confirm important "
                "information.",
                "My answers can be incomplete or out of date, and browsing is "
                "off by default. Use me as a first draft, not a final "
                "authority.",
            ),
            ("memory", "context", "answers", "draft"),
        ),
    ),
    "greeting": (
        _group(
            (
                "Hello!",
                "Hi there!",
                "Hey, good to see you.",
            ),
            (
                "Hello! It is good to meet you. What can I help you with "
                "today?",
                "Hi! I'm ready to help, so tell me what you are working on.",
                "Hey! Ask me anything and I will help with a short, useful "
                "answer.",
            ),
            ("hello", "hi", "hey"),
        ),
        _group(
            (
                "How are you?",
                "How is your day going?",
                "How are you doing today?",
            ),
            (
                "I'm doing well and ready to help. What would you like to work "
                "on?",
                "All good here, thanks for asking. What can I do for you "
                "today?",
                "I'm running smoothly and glad to chat. How can I help?",
            ),
            ("doing well", "all good", "running smoothly"),
        ),
        _group(
            (
                "Good morning!",
                "Good morning, how are you?",
                "Morning! I hope you slept well.",
            ),
            (
                "Good morning! I hope your day is off to a good start. What "
                "can I help with?",
                "Good morning! Ready when you are; what are we working on "
                "today?",
                "Morning! Let's make the day easier. What would you like to "
                "start with?",
            ),
            ("morning",),
        ),
        _group(
            (
                "Good evening!",
                "Good evening, how is it going?",
                "Evening! I hope you had a good day.",
            ),
            (
                "Good evening! I hope your day went well. What would you like "
                "to work on?",
                "Good evening! I'm here and ready to help with a question or "
                "a quick task.",
                "Evening! Let's keep it simple; tell me what you need and I "
                "will take a look.",
            ),
            ("evening", "help"),
        ),
        _group(
            (
                "Thank you!",
                "Thanks, that was helpful.",
                "Thanks for the quick answer.",
            ),
            (
                "You're welcome! I'm glad it helped. Ask again whenever you "
                "need another hand.",
                "Happy to help! If anything is unclear, I can expand on it.",
                "You're welcome. A quick follow-up question is always fine.",
            ),
            ("welcome", "help"),
        ),
        _group(
            (
                "Goodbye!",
                "See you later.",
                "I have to go now.",
            ),
            (
                "Goodbye! I'll be here when you need another answer.",
                "See you later! Save your notes and pick up where you left "
                "off.",
                "Take care! Come back anytime you want to continue.",
            ),
            ("goodbye", "later", "care"),
        ),
        _group(
            (
                "Nice to meet you.",
                "It's a pleasure to meet you.",
                "Pleased to meet you!",
            ),
            (
                "Nice to meet you too! Tell me what you are working on and I "
                "will help.",
                "The pleasure is mine. What would you like to start with?",
                "Glad to meet you! I'm ready for your first question.",
            ),
            ("meet", "pleasure", "glad"),
        ),
        _group(
            (
                "How's it going?",
                "What's up?",
                "How is everything?",
            ),
            (
                "Things are going well here. What can I help you with?",
                "Not much on my side; I'm ready to help. What do you need?",
                "All steady here. Tell me what you would like to get done.",
            ),
            ("well", "help", "steady"),
        ),
    ),
    "explanation": (
        _group(
            (
                "What is a transformer?",
                "Can you explain transformer models simply?",
                "How does a transformer neural network work?",
            ),
            (
                "A transformer is a neural network architecture built around "
                "attention. It lets a model weigh the parts of an input that "
                "matter most and process tokens in parallel.",
                "A transformer is a model design that uses self-attention to "
                "relate words or tokens to each other. That makes it effective "
                "for language tasks.",
                "In simple terms, a transformer learns which parts of the "
                "input to focus on through attention layers. Most modern "
                "language models use this architecture.",
            ),
            ("attention",),
        ),
        _group(
            (
                "What is RAG?",
                "What does retrieval-augmented generation mean?",
                "How does retrieval-augmented generation work?",
            ),
            (
                "RAG stands for retrieval-augmented generation. It retrieves "
                "relevant documents first, then gives them to the model as "
                "context for the answer.",
                "Retrieval-augmented generation combines a search step with "
                "text generation. The model answers using passages fetched "
                "from a document store.",
                "RAG means the system looks up relevant sources before "
                "answering. That grounds the reply in retrieved text instead "
                "of relying only on model memory.",
            ),
            ("retriev",),
        ),
        _group(
            (
                "What is a database?",
                "Can you explain what a database is?",
                "How would you describe a database to a beginner?",
            ),
            (
                "A database is an organized collection of data that can be "
                "stored, searched, and updated. A database management system "
                "handles those operations.",
                "A database stores structured information so it can be queried "
                "reliably. Examples include tables of users, orders, or "
                "inventory.",
                "Think of a database as a structured store for records. "
                "Software such as a DBMS lets applications read, write, and "
                "update those records safely.",
            ),
            ("database",),
        ),
        _group(
            (
                "What is an API?",
                "Can you explain APIs simply?",
                "How does an API work?",
            ),
            (
                "An API is a defined way for one program to request data or "
                "actions from another. It sets the names, inputs, and response "
                "format.",
                "API stands for application programming interface. It is a "
                "contract that lets software components talk without knowing "
                "each other's internals.",
                "Think of an API as a menu of operations a service exposes. A "
                "program sends a request, and the service returns a "
                "structured response.",
            ),
            ("api",),
        ),
        _group(
            (
                "What is a cache?",
                "Can you explain caching simply?",
                "Why do programs use a cache?",
            ),
            (
                "A cache is a fast store for data that is expensive to fetch "
                "or compute. It trades a little memory for much lower "
                "latency.",
                "Caching keeps a copy of frequently used data close at hand. "
                "A good cache has a limit and a rule for removing old "
                "entries.",
                "A cache speeds up repeated work by remembering an earlier "
                "result. The hard part is deciding when that copy is stale.",
            ),
            ("cache",),
        ),
        _group(
            (
                "What is version control?",
                "Can you explain Git simply?",
                "Why should I use version control?",
            ),
            (
                "Version control records changes to files so you can review, "
                "compare, and restore earlier versions. Git is the most "
                "common tool for it.",
                "A version control system keeps a history of edits and lets "
                "several people work on the same project. Git stores "
                "snapshots called commits.",
                "Version control gives you a timeline of your work and a safe "
                "way to undo mistakes. Commit often so each change is easy to "
                "find.",
            ),
            ("version control", "git", "commit"),
        ),
        _group(
            (
                "What is encryption?",
                "Can you explain encryption simply?",
                "How does encryption protect data?",
            ),
            (
                "Encryption turns readable data into ciphertext using a key. "
                "Only someone with the right key can turn it back into "
                "plaintext.",
                "Encryption scrambles information so it looks meaningless "
                "without the decryption key. It protects data in transit and "
                "at rest.",
                "At a high level, encryption is a lock and a key for data. "
                "Modern systems use it for messages, storage, and secure "
                "connections.",
            ),
            ("encrypt", "key", "ciphertext"),
        ),
        _group(
            (
                "What is JSON?",
                "Can you explain the JSON format?",
                "Why is JSON used for data exchange?",
            ),
            (
                "JSON is a text format for structured data using objects, "
                "arrays, strings, numbers, booleans, and null. It is easy for "
                "both people and programs to read.",
                "JSON stands for JavaScript Object Notation, but many "
                "languages can parse it. It is common in APIs because it is "
                "compact and simple.",
                "JSON represents data as key-value pairs and ordered lists. "
                "It works well for configuration files and network "
                "responses.",
            ),
            ("json",),
        ),
    ),
    "cooking": (
        _group(
            (
                "How do I boil an egg?",
                "What is the best way to make a hard-boiled egg?",
                "How long should I cook an egg in boiling water?",
            ),
            (
                "Lower eggs into boiling water and cook for about 9 to 10 "
                "minutes for a firm yolk. Move them to cold water afterward so "
                "they are easy to peel.",
                "For a hard-boiled egg, simmer for roughly 10 minutes, then "
                "cool it in ice water. For a softer yolk, take it out at 6 to "
                "7 minutes.",
                "Place the eggs in a pot of boiling water, cook 9 to 10 "
                "minutes, and transfer them to an ice bath. That gives a firm "
                "yolk and clean peeling.",
            ),
            ("boil",),
        ),
        _group(
            (
                "How do I make simple pasta?",
                "What is an easy pasta recipe for a beginner?",
                "Can you give me a quick pasta method?",
            ),
            (
                "Boil salted water, add the pasta, and cook it according to the "
                "package time. Drain it, then toss with olive oil, garlic, and "
                "grated cheese.",
                "Cook pasta in well-salted boiling water until just tender. "
                "Save a little cooking water, mix it with your sauce, and "
                "combine everything in the pot.",
                "For a quick meal, boil pasta, reserve a cup of the water, and "
                "drain. Stir the water into a simple tomato or butter sauce so "
                "it coats the pasta.",
            ),
            ("pasta",),
        ),
        _group(
            (
                "How should I store bread?",
                "What is the best way to keep bread fresh?",
                "How do I store a loaf of bread so it lasts longer?",
            ),
            (
                "Keep bread at room temperature in a paper bag or bread box "
                "for two to three days. For longer storage, slice it and freeze "
                "it.",
                "A paper bag keeps the crust from getting soggy, and it works "
                "well for a few days. Freeze extra slices and toast them when "
                "needed.",
                "Store everyday bread in a cool, dry place and avoid the "
                "refrigerator. Freeze what you will not finish within a few "
                "days.",
            ),
            ("bread", "freeze"),
        ),
        _group(
            (
                "How do I cook white rice?",
                "What is a simple way to make rice?",
                "How much water should I use for rice?",
            ),
            (
                "Rinse the rice, then simmer one part rice with about two "
                "parts water for 15 to 18 minutes. Rest it covered for five "
                "minutes before fluffing.",
                "Use a 1 to 2 ratio of rice to water, bring it to a boil, "
                "then cover and cook on low until the water is absorbed. Let "
                "it stand before serving.",
                "For a simple pot of rice, rinse it, add twice its volume of "
                "water, and simmer covered until tender. Turn off the heat "
                "and let it steam.",
            ),
            ("rice",),
        ),
        _group(
            (
                "How do I roast vegetables?",
                "What is a simple vegetable roasting method?",
                "How long should I roast vegetables?",
            ),
            (
                "Cut the vegetables into similar-sized pieces, toss them with "
                "oil and salt, and roast at about 220 degrees Celsius until "
                "browned at the edges.",
                "Spread vegetables in a single layer on a tray so they brown "
                "instead of steam. Roast them hot for 20 to 30 minutes, "
                "turning once.",
                "Coat vegetables lightly in oil, season them, and give them "
                "space on the pan. Check them after 20 minutes and add time "
                "if needed.",
            ),
            ("roast", "vegetable", "tray"),
        ),
        _group(
            (
                "How do I make a good cup of tea?",
                "What is the right way to brew tea?",
                "How long should tea steep?",
            ),
            (
                "Use freshly boiled water and the amount of tea suggested on "
                "the packet. Steep black tea for about three to five minutes, "
                "then remove the leaves.",
                "Warm the cup, add tea, and pour hot water over it. Green tea "
                "usually needs cooler water and a shorter steep than black "
                "tea.",
                "Start with one teaspoon of loose tea per cup and steep to "
                "taste. Remove the leaves when the flavor is strong enough to "
                "avoid bitterness.",
            ),
            ("tea", "steep"),
        ),
        _group(
            (
                "How do I make simple pancakes?",
                "What is an easy pancake recipe?",
                "Can you give me a basic pancake method?",
            ),
            (
                "Mix flour, milk, eggs, and a pinch of salt into a smooth "
                "batter. Cook small ladles in a hot, lightly oiled pan until "
                "bubbles form, then flip.",
                "For basic pancakes, combine one cup of flour with an egg and "
                "enough milk for a pourable batter. Rest the batter for ten "
                "minutes before cooking.",
                "Whisk flour, baking powder, milk, and an egg, then let the "
                "batter sit briefly. Pour it into a warm pan and turn each "
                "pancake when the top sets.",
            ),
            ("pancake", "batter", "flour"),
        ),
        _group(
            (
                "How do I make a simple salad dressing?",
                "What is a basic vinaigrette recipe?",
                "Can you give me an easy dressing idea?",
            ),
            (
                "Whisk three parts olive oil with one part vinegar, then add "
                "salt, pepper, and a little mustard. Taste and adjust the "
                "balance before serving.",
                "A simple vinaigrette is oil, vinegar, salt, and pepper "
                "shaken together. Add honey for sweetness or mustard to help "
                "it emulsify.",
                "Mix olive oil, lemon juice, salt, and a spoon of mustard in "
                "a small jar. Shake it well and drizzle it over the salad.",
            ),
            ("oil", "vinegar", "dressing"),
        ),
    ),
    "travel": (
        _group(
            (
                "How do I pack a carry-on bag?",
                "What should I put in a carry-on for a short trip?",
                "Can you help me pack light for a flight?",
            ),
            (
                "Start with a small capsule wardrobe, then add chargers, "
                "medications, and a clear toiletries bag. Check your airline's "
                "size and liquid limits before packing.",
                "Roll clothes to save space in a carry-on, wear the bulkiest "
                "shoes, and keep documents and medication in one easy-to-reach "
                "pouch. Leave room for anything you buy.",
                "Pack two or three mix-and-match outfits, a charger, and only "
                "the toiletries you need. Confirm the carry-on dimensions and "
                "security rules for your airline.",
            ),
            ("pack", "carry-on"),
        ),
        _group(
            (
                "How should I plan a day trip?",
                "What should I bring on a day trip?",
                "Can you help me plan a one-day outing?",
            ),
            (
                "Choose a destination within a comfortable travel time and "
                "check the weather first. Bring water, snacks, a charged "
                "phone, and a plan for the return trip.",
                "Pick two or three main stops, then leave space between them. "
                "Carry water, a light layer, and a map or offline directions.",
                "Plan your route, book anything that needs a reservation, and "
                "set a return time. Pack water, snacks, and a portable "
                "charger.",
            ),
            ("water", "trip"),
        ),
        _group(
            (
                "How early should I arrive at the airport?",
                "When should I get to the airport for a flight?",
                "What is a safe airport arrival time?",
            ),
            (
                "Arrive about two hours before a domestic flight and three "
                "hours before an international one. Add extra time for checked "
                "bags or busy holidays.",
                "A good rule is two hours for domestic departures and three "
                "for international flights. Check your airline's guidance "
                "because security lines vary.",
                "Plan to reach the terminal two to three hours early, "
                "depending on the route. That leaves time for check-in, "
                "security, and finding the gate.",
            ),
            ("hour",),
        ),
        _group(
            (
                "How do I reduce jet lag?",
                "What helps with jet lag after a long flight?",
                "How can I adjust to a new time zone?",
            ),
            (
                "Adjust your sleep schedule a little before departure and get "
                "daylight at the new destination. Keep meals and bedtime "
                "close to local time.",
                "After a long flight, stay awake until a normal local bedtime "
                "and spend time outdoors. Daylight is the strongest signal "
                "for resetting your body clock.",
                "Hydrate, avoid heavy late meals, and expose yourself to "
                "morning light for the first few days. Short naps are fine if "
                "they stay under 30 minutes.",
            ),
            ("jet lag", "local", "daylight", "light"),
        ),
        _group(
            (
                "How can I save money on a trip?",
                "What is a good travel budget?",
                "How do I plan a trip budget?",
            ),
            (
                "Set a total budget, then split it across transport, lodging, "
                "food, and activities. Track daily spending so one category "
                "does not quietly take over.",
                "Book transport and lodging early, travel outside peak dates "
                "when possible, and keep a small buffer for unexpected "
                "costs.",
                "Start with fixed costs such as travel and accommodation, "
                "then add a daily allowance for meals and entry fees. A ten "
                "percent buffer covers surprises.",
            ),
            ("budget", "cost", "allowance"),
        ),
        _group(
            (
                "How do I use public transit in a new city?",
                "What should I know about getting around a new city?",
                "How can I navigate a city without a car?",
            ),
            (
                "Check the local transit app or station map, buy a day pass "
                "if you will ride often, and keep the route saved offline. "
                "Allow extra time during rush hour.",
                "Learn the main subway or bus lines near your lodging and "
                "confirm the fare before boarding. A stored card or "
                "contactless payment is usually fastest.",
                "Plan one route in advance and note the return stop. Ask "
                "staff at the station if the signs are unclear, and avoid "
                "rushing at busy hours.",
            ),
            ("transit", "route", "station", "bus", "subway"),
        ),
        _group(
            (
                "What travel documents should I check?",
                "How do I prepare my passport and papers for a trip?",
                "What should I do before an international trip?",
            ),
            (
                "Check that your passport is valid for the whole trip and any "
                "required visa is approved. Store digital and paper copies "
                "separately from the original.",
                "Review entry rules for each country, confirm the name on "
                "your ticket matches your passport, and note your embassy "
                "contact. Keep copies of key documents.",
                "Before departure, verify passport validity, visas, and "
                "insurance details. Put copies in a bag and share the "
                "itinerary with someone you trust.",
            ),
            ("passport", "visa", "document"),
        ),
        _group(
            (
                "How do I choose a good hotel?",
                "What should I check before booking a hotel?",
                "How can I get a better hotel room?",
            ),
            (
                "Compare the location, total price with taxes, and recent "
                "reviews. A central room can save time even if it costs a "
                "little more.",
                "Book a refundable rate when plans are uncertain, and confirm "
                "check-in times and cancellation rules for the booking. Ask "
                "for a quiet room away from the elevator.",
                "Check the room size, bed type, and distance to transit "
                "before booking. A direct reservation sometimes gives more "
                "flexibility than a discount site.",
            ),
            ("hotel", "room", "booking"),
        ),
    ),
    "coding": (
        _group(
            (
                "What is a function in programming?",
                "Can you explain functions in code?",
                "How do functions work in Python?",
            ),
            (
                "A function is a named block of code that takes inputs, "
                "performs a task, and usually returns a result. You call it by "
                "name whenever you need that task.",
                "A function groups reusable logic behind a name. Parameters "
                "pass data in, and a return statement sends a result back.",
                "In Python, a function is defined with def and can accept "
                "arguments. It keeps code organized and avoids repeating the "
                "same steps.",
            ),
            ("function",),
        ),
        _group(
            (
                "How do I debug a typo in my code?",
                "My program has a spelling mistake; how do I find it?",
                "What is the fastest way to fix a typo in code?",
            ),
            (
                "Read the error message and go to the line it names. Check "
                "variable and function names character by character, then run "
                "the smallest test that reproduces it.",
                "Search the file for the name in the error and compare it with "
                "where the variable was defined. An editor's rename or find "
                "feature makes mismatches easier to spot.",
                "Start at the reported line, check capitalization and "
                "spelling, and print the value before it fails. A quick rerun "
                "confirms the fix.",
            ),
            ("error", "spelling", "typo"),
        ),
        _group(
            (
                "Can you explain loops in programming?",
                "What is a loop in code?",
                "How does a for loop work?",
            ),
            (
                "A loop repeats a block of code until a condition is met or a "
                "collection is finished. A for loop visits each item, while a "
                "while loop checks a condition before each pass.",
                "Loops let you avoid writing the same instruction many times. "
                "For example, a for loop can process every item in a list.",
                "A loop runs the same steps repeatedly. Use a for loop for a "
                "known sequence and a while loop when the stopping condition "
                "is dynamic.",
            ),
            ("loop",),
        ),
        _group(
            (
                "What is the difference between a list and a dictionary?",
                "When should I use a list instead of a dictionary?",
                "Can you explain lists and dictionaries?",
            ),
            (
                "A list stores items in order and is best when position "
                "matters. A dictionary stores values under keys and is best "
                "for fast lookup by name.",
                "Use a list for a sequence and a dictionary for labeled data. "
                "Lists keep order, while dictionaries map each key to one "
                "value.",
                "Lists are indexed by number and dictionaries by key. If you "
                "need to find an item by name, a dictionary is usually "
                "clearer.",
            ),
            ("list", "dictionary"),
        ),
        _group(
            (
                "How do I make a good git commit?",
                "What should a commit message include?",
                "How should I organize commits?",
            ),
            (
                "Keep each commit focused on one change and write a short "
                "message that says what it does. Add a body only when the "
                "reason is not obvious.",
                "Stage only the files for the current change, then describe "
                "the result in the subject line. Small commits are easier to "
                "review and revert.",
                "A good commit is small, tested, and named for its effect, "
                "such as add input validation. Avoid mixing formatting "
                "changes with behavior changes.",
            ),
            ("commit", "change"),
        ),
        _group(
            (
                "How do I write a simple unit test?",
                "What makes a good software test?",
                "How do I start testing my code?",
            ),
            (
                "Pick one small behavior, give it a known input, and assert "
                "the expected output. Run the test before and after the "
                "change.",
                "A good test is fast, repeatable, and focused on one behavior. "
                "Start with the most important path and add edge cases later.",
                "Write a test that fails for the bug, then make it pass with "
                "the smallest change. Keep the test independent of other "
                "tests.",
            ),
            ("test",),
        ),
        _group(
            (
                "How should I name variables?",
                "What makes a good variable name?",
                "How do I make code easier to read?",
            ),
            (
                "Use descriptive names that say what the value means, not its "
                "type. Short names are fine only for tiny, local pieces of "
                "code.",
                "Choose names a new reader can understand without extra "
                "context. Prefer full words over abbreviations unless the "
                "abbreviation is standard.",
                "Name variables for their purpose, keep one concept per name, "
                "and be consistent. Clear names reduce the need for comments.",
            ),
            ("name", "read", "code"),
        ),
        _group(
            (
                "How do I read a Python traceback?",
                "What should I look for in an error message?",
                "How do I find the cause of an error?",
            ),
            (
                "Read the last line first for the error type and message, "
                "then follow the file and line numbers upward. The deepest "
                "frame is usually where it failed.",
                "A traceback shows the call path from the outer function to "
                "the failure. Start at the line in your own code that appears "
                "last.",
                "Find the exception name and the line that raised it, then "
                "check the values at that point. A small print or debugger "
                "session usually explains it.",
            ),
            ("traceback", "error", "exception", "line"),
        ),
    ),
    "planning": (
        _group(
            (
                "How should I plan a study week?",
                "Can you help me make a weekly study plan?",
                "What is a good study schedule for one week?",
            ),
            (
                "List the study topics you need to cover and split them into "
                "small daily blocks. Add short reviews, a buffer day, and "
                "enough sleep.",
                "Start with your deadlines, then assign two focused study "
                "blocks per day. End the week with a practice test or a summary "
                "review.",
                "Choose three main study goals for the week and give each one "
                "a specific time slot. Include breaks and one catch-up "
                "session.",
            ),
            ("study",),
        ),
        _group(
            (
                "How do I prioritize my tasks?",
                "What is a simple way to decide what to do first?",
                "How can I sort tasks by priority?",
            ),
            (
                "Write down every task, mark which are urgent and which are "
                "important, and do the important ones first. Schedule or drop "
                "the rest.",
                "Use a short list and rank items by impact and deadline. Start "
                "with the task that unblocks the most other work.",
                "Group tasks into do now, schedule, delegate, and drop. Then "
                "focus on one item at a time.",
            ),
            ("task", "item"),
        ),
        _group(
            (
                "How do I make a good checklist?",
                "What should a useful checklist include?",
                "Can you help me build a task checklist?",
            ),
            (
                "Start a checklist with the goal, list the steps in the order "
                "they happen, and add an owner or deadline for each. Keep it "
                "short enough to use.",
                "A good checklist has clear, actionable items and a way to mark "
                "them done. Put the most important step first.",
                "Write each item as a single action, group related steps, and "
                "include a final check. Review the list before you start.",
            ),
            ("goal", "step", "item"),
        ),
        _group(
            (
                "How do I build a good morning routine?",
                "What should a simple morning routine include?",
                "How can I start my day well?",
            ),
            (
                "Choose two or three actions that matter, such as water, a "
                "short walk, and a review of the day's top task. Keep the "
                "routine short enough to repeat.",
                "A simple routine might include a fixed wake time, breakfast, "
                "and five minutes of planning. Put the most important step "
                "first.",
                "Start with one habit you can do consistently, then add "
                "another after a week. Keep the phone out of the first few "
                "minutes if it distracts you.",
            ),
            ("routine", "morning", "day", "habit"),
        ),
        _group(
            (
                "How do I break a project into steps?",
                "How do I plan a small project?",
                "What is a simple project plan?",
            ),
            (
                "Define the outcome, list the main deliverables, and split "
                "each one into tasks that take a day or less. Order them by "
                "dependency.",
                "Start with the finish line and work backward into milestones. "
                "Give each milestone a date, an owner, and a clear definition "
                "of done.",
                "Write a one-page plan with scope, milestones, risks, and the "
                "next action. Review it weekly and update it when assumptions "
                "change.",
            ),
            ("project", "milestone", "task", "plan"),
        ),
        _group(
            (
                "How do I write a meeting agenda?",
                "What should a good meeting agenda include?",
                "How do I keep a meeting focused?",
            ),
            (
                "List the decisions or outcomes the meeting must produce, "
                "then give each item a time box and an owner. Send the agenda "
                "in advance.",
                "A useful agenda states the goal, the topics in order, and "
                "who leads each one. Reserve the last few minutes for actions "
                "and owners.",
                "Start with the most important item and limit the list to "
                "what fits the time. Close by reading back the decisions and "
                "next steps.",
            ),
            ("agenda", "meeting", "decision"),
        ),
        _group(
            (
                "How do I do a weekly review?",
                "What should I include in a weekly review?",
                "How can I review my week effectively?",
            ),
            (
                "Set aside 20 minutes to review what you finished, what "
                "slipped, and what matters next week. Choose a small number "
                "of priorities.",
                "Look back at your calendar and task list, note patterns, and "
                "clear loose ends. Then pick the top three outcomes for the "
                "coming week.",
                "At the end of the week, ask three questions: what went well, "
                "what got stuck, and what changes next. Write the answers and "
                "schedule the first action.",
            ),
            ("week", "review", "priority"),
        ),
        _group(
            (
                "How do I build a new habit?",
                "What is the best way to start a habit?",
                "How long does it take to form a habit?",
            ),
            (
                "Start with a version so small it is easy to do daily, and "
                "attach it to an existing routine. Track the streak and "
                "increase the size later.",
                "Choose one habit, define a clear cue, and make the first "
                "version take under five minutes. Consistency matters more "
                "than intensity at the start.",
                "Expect weeks rather than days for a habit to feel automatic. "
                "Missing once is fine; aim to return to the routine the next "
                "day.",
            ),
            ("habit", "routine", "daily"),
        ),
    ),
    "general_knowledge": (
        _group(
            (
                "What is the capital of France?",
                "Which city is the capital of France?",
                "Can you tell me France's capital?",
            ),
            (
                "The capital of France is Paris, the country's largest city and "
                "main cultural center.",
                "Paris is the capital of France and its largest city by "
                "population.",
                "France's capital city is Paris, located in the north-central "
                "part of the country.",
            ),
            ("paris",),
        ),
        _group(
            (
                "What is the boiling point of water?",
                "At what temperature does water boil at sea level?",
                "How hot does water need to be to boil?",
            ),
            (
                "At sea level, water boils at 100 degrees Celsius, which is 212 "
                "degrees Fahrenheit.",
                "Water boils at about 100 degrees Celsius at standard "
                "atmospheric pressure. At higher altitudes, the boiling point "
                "is lower.",
                "The boiling point of water is 100 degrees Celsius or 212 "
                "degrees Fahrenheit at sea level.",
            ),
            ("100",),
        ),
        _group(
            (
                "What is the speed of light?",
                "How fast does light travel in a vacuum?",
                "Can you tell me the speed of light?",
            ),
            (
                "Light travels at about 299,792 kilometers per second in a "
                "vacuum, or roughly 186,282 miles per second.",
                "The speed of light in a vacuum is approximately 300,000 "
                "kilometers per second.",
                "In a vacuum, light moves at 299,792,458 meters per second, "
                "which is about 186,000 miles per second.",
            ),
            ("299,792", "300,000", "186,000"),
        ),
        _group(
            (
                "What is the largest planet in the solar system?",
                "Which planet is the biggest?",
                "Can you tell me the largest planet?",
            ),
            (
                "Jupiter is the largest planet in the solar system. It is a "
                "gas giant with a mass greater than all the other planets "
                "combined.",
                "The largest planet is Jupiter, followed by Saturn. Jupiter "
                "is about eleven times wider than Earth.",
                "Jupiter holds the title of largest planet. Its Great Red "
                "Spot is a long-lived storm larger than Earth.",
            ),
            ("jupiter",),
        ),
        _group(
            (
                "What is the largest ocean on Earth?",
                "Which ocean is the biggest?",
                "Can you name the largest ocean?",
            ),
            (
                "The Pacific Ocean is the largest and deepest ocean on Earth. "
                "It covers about one third of the planet's surface.",
                "The Pacific is the biggest ocean, stretching between Asia "
                "and the Americas. It contains the deepest known trench.",
                "Earth's largest ocean is the Pacific. It is larger than all "
                "the other oceans combined.",
            ),
            ("pacific", "ocean"),
        ),
        _group(
            (
                "How many continents are there?",
                "Can you list the continents?",
                "What are the seven continents?",
            ),
            (
                "There are seven continents: Africa, Antarctica, Asia, "
                "Europe, North America, Oceania, and South America.",
                "The usual count is seven continents, including Asia, Africa, "
                "Europe, and Antarctica. North America, South America, and "
                "Oceania complete the list.",
                "Most school systems teach seven continents. The exact "
                "grouping can vary, but Asia is the largest by area and "
                "population.",
            ),
            ("continent", "seven"),
        ),
        _group(
            (
                "What is photosynthesis?",
                "How does photosynthesis work?",
                "Why do plants need sunlight?",
            ),
            (
                "Photosynthesis is how plants use sunlight to turn carbon "
                "dioxide and water into sugar and oxygen. Chlorophyll in the "
                "leaves captures the light.",
                "In photosynthesis, plants convert light energy into chemical "
                "energy. The process releases oxygen and stores energy in "
                "glucose.",
                "Plants need sunlight because it powers photosynthesis. "
                "Without enough light, they struggle to make the sugar they "
                "use for growth.",
            ),
            ("photosynthesis", "oxygen", "glucose"),
        ),
        _group(
            (
                "Why is the sky blue?",
                "What makes the sky look blue?",
                "Can you explain why the sky is blue?",
            ),
            (
                "Sunlight contains many colors, and air molecules scatter "
                "blue light more than red. That scattered blue light reaches "
                "your eyes from every direction.",
                "The sky looks blue because of Rayleigh scattering. Shorter "
                "wavelengths scatter more strongly in the atmosphere, so blue "
                "dominates during the day.",
                "Air scatters the short blue wavelengths of sunlight across "
                "the sky. At sunset, the light travels farther and reds and "
                "oranges become more visible.",
            ),
            ("blue", "scatter", "rayleigh"),
        ),
    ),
}


def normalize_text(value: Any) -> str:
    """Case-fold and collapse whitespace for narrow lexical comparison."""

    return " ".join(
        str(value or "").replace("\u2019", "'").split()).casefold()


def sentence_count(value: str) -> int:
    """Count sentence-ending punctuation followed by whitespace or the end."""

    return len(_SENTENCE_END_RE.findall(value.strip()))


def _contains_personal_data(value: str) -> bool:
    return bool(
        _EMAIL_RE.search(value)
        or _URL_RE.search(value)
        or _PHONE_RE.search(value))


def validate_user(value: Any) -> bool:
    """Return True iff ``value`` is a short, single-line prompt string."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\n" in value or "\r" in value:
        return False
    if len(text) > MAX_USER_CHARS:
        return False
    return True


def validate_answer(value: Any) -> bool:
    """Return True iff ``value`` is a short, safe, non-refusal answer."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\n" in value or "\r" in value:
        return False
    if not MIN_ANSWER_CHARS <= len(text) <= MAX_ANSWER_CHARS:
        return False
    if not 1 <= sentence_count(text) <= 3:
        return False
    reply = normalize_text(text)
    if any(marker in reply for marker in REFUSAL_MARKERS):
        return False
    if any(term in reply for term in UNSAFE_TERMS):
        return False
    if _contains_personal_data(text):
        return False
    return True


def validate_row(row: Mapping[str, Any]) -> bool:
    """Return True iff ``row`` has the benign chat schema and invariants."""

    if set(row) != set(REQUIRED_ROW_FIELDS):
        return False
    if row.get("family") not in CATEGORY_ORDER:
        return False
    if row.get("source") != SOURCE:
        return False
    if row.get("style") != STYLE:
        return False
    if not validate_user(row.get("user")):
        return False
    if not validate_answer(row.get("assistant")):
        return False
    return True


def _validate_group(category: str, group_index: int,
                    group: Mapping[str, Any]) -> None:
    prompts = group.get("prompts")
    answers = group.get("answers")
    concrete = group.get("concrete")
    if (not isinstance(prompts, tuple) or len(prompts) < 2
            or any(not validate_user(prompt) for prompt in prompts)):
        raise ValueError(
            f"{category} group {group_index} has invalid prompt templates")
    if len(set(prompts)) != len(prompts):
        raise ValueError(
            f"{category} group {group_index} repeats a prompt template")
    if (not isinstance(answers, tuple) or len(answers) < 2
            or any(not validate_answer(answer) for answer in answers)):
        raise ValueError(
            f"{category} group {group_index} has invalid answer templates")
    if len(set(answers)) != len(answers):
        raise ValueError(
            f"{category} group {group_index} repeats an answer template")
    if (not isinstance(concrete, tuple) or not concrete
            or any(not isinstance(term, str) or not term.strip()
                   for term in concrete)):
        raise ValueError(
            f"{category} group {group_index} has invalid concrete terms")
    for answer in answers:
        reply = normalize_text(answer)
        if not any(normalize_text(term) in reply for term in concrete):
            raise ValueError(
                f"{category} group {group_index} answer lacks a concrete "
                f"term: {answer!r}")


def _combinations(category: str) -> list[tuple[str, str]]:
    if category not in TEMPLATES:
        raise ValueError(f"missing template category {category!r}")
    combinations: list[tuple[str, str]] = []
    for group_index, group in enumerate(TEMPLATES[category]):
        _validate_group(category, group_index, group)
        for prompt in group["prompts"]:
            for answer in group["answers"]:
                combinations.append((prompt, answer))
    if len(set(combinations)) != len(combinations):
        raise ValueError(f"{category} template combinations are not unique")
    return combinations


def build_rows(*, per_category: int = DEFAULT_PER_CATEGORY,
               seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Build the deterministic benign chat rows for one seed.

    The same ``seed`` and ``per_category`` always produce the same rows in the
    same order. A different seed selects a different combination subset when
    the pool is larger than ``per_category``. Every normalized
    ``(user, assistant)`` pair is unique.
    """

    if isinstance(per_category, bool) or not isinstance(per_category, int):
        raise ValueError("per_category must be an int")
    if per_category < 1:
        raise ValueError("per_category must be at least 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")

    pools = {
        category: _combinations(category) for category in CATEGORY_ORDER
    }
    pool_keys: set[tuple[str, str]] = set()
    for category in CATEGORY_ORDER:
        for user, assistant in pools[category]:
            key = (normalize_text(user), normalize_text(assistant))
            if key in pool_keys:
                raise ValueError(
                    "duplicate (user, assistant) pair in the template pool")
            pool_keys.add(key)

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for category_index, category in enumerate(CATEGORY_ORDER):
        combinations = pools[category]
        if per_category > len(combinations):
            raise ValueError(
                f"category {category!r} has only {len(combinations)} "
                f"template combinations; per_category {per_category} is too "
                f"large")
        indices = list(range(len(combinations)))
        random.Random(seed + category_index * 100003).shuffle(indices)
        chosen = sorted(indices[:per_category])
        for index in chosen:
            user, assistant = combinations[index]
            row = {
                "user": user,
                "assistant": assistant,
                "family": category,
                "source": SOURCE,
                "style": STYLE,
            }
            if not validate_row(row):
                raise ValueError(
                    f"invalid {category!r} row at combination {index}")
            key = (normalize_text(user), normalize_text(assistant))
            if key in seen:
                raise ValueError(
                    f"duplicate (user, assistant) pair in {category!r}")
            seen.add(key)
            rows.append(row)
    if len(rows) != per_category * len(CATEGORY_ORDER):
        raise ValueError(
            f"expected {per_category * len(CATEGORY_ORDER)} rows, "
            f"built {len(rows)}")
    return rows


def render_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render rows as UTF-8 JSONL text with a trailing newline."""

    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def build_manifest(rows: Sequence[Mapping[str, Any]], *,
                   seed: int = DEFAULT_SEED,
                   per_category: int = DEFAULT_PER_CATEGORY) -> dict[str, Any]:
    """Build a content-free manifest for ``rows``.

    Only aggregate counts, the seed, configuration and a hash are included.
    No prompt or answer text is copied into the manifest.
    """

    content = render_jsonl(rows).encode("utf-8")
    counts = Counter(row["family"] for row in rows)
    return {
        "schema": SCHEMA_VERSION,
        "source": SOURCE,
        "style": STYLE,
        "seed": int(seed),
        "per_category": int(per_category),
        "total": len(rows),
        "categories": list(CATEGORY_ORDER),
        "counts": {
            "by_category": {
                category: counts.get(category, 0)
                for category in CATEGORY_ORDER
            },
        },
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "content_free": True,
    }


def write_dataset(rows: Sequence[Mapping[str, Any]],
                  out: str | Path = DEFAULT_OUT, *,
                  manifest: str | Path = DEFAULT_MANIFEST,
                  seed: int = DEFAULT_SEED,
                  per_category: int = DEFAULT_PER_CATEGORY,
                  force: bool = False) -> dict[str, Any]:
    """Write the benign chat JSONL and manifest.

    Existing output or manifest files are never overwritten unless ``force``
    is true. Nothing is written when either path already exists.
    """

    out_path = Path(out).expanduser()
    manifest_path = Path(manifest).expanduser()
    if out_path.resolve() == manifest_path.resolve():
        raise ValueError("--out and --manifest must be different files")
    for path, label in ((out_path, "output"), (manifest_path, "manifest")):
        if path.exists() and not force:
            raise FileExistsError(
                f"{label} already exists: {path}; use --force to overwrite")
    content = render_jsonl(rows)
    summary = build_manifest(
        rows, seed=seed, per_category=per_category)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content.encode("utf-8"))
    manifest_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        newline="\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"training JSONL path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST,
        help=f"content-free manifest path (default: {DEFAULT_MANIFEST})")
    parser.add_argument(
        "--per-category", type=int, default=DEFAULT_PER_CATEGORY,
        help=f"rows per category (default: {DEFAULT_PER_CATEGORY})")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"deterministic RNG seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing output and manifest files")
    args = parser.parse_args(argv)
    try:
        rows = build_rows(
            per_category=args.per_category, seed=args.seed)
        summary = write_dataset(
            rows, args.out, manifest=args.manifest, seed=args.seed,
            per_category=args.per_category, force=args.force)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
