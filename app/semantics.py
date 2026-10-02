"""Language-independent meaning layer. A concept ID is the meaning; each language has its OWN realization.
English is only a display label and a speech-matching alias, never a pivot.
All realizations below are machine-authored drafts: have a native speaker review them before claiming support."""
import re

# concept -> kind (used by clarification questions)
CONCEPTS = {
    "bank": "place", "storeorshop": "place",
    "black": "colour", "blue": "colour", "red": "colour", "white": "colour",
    "shoes": "clothing", "tshirt": "clothing",
    "cellphone": "object", "pen": "object",
    "biglarge": "size", "smalllittle": "size",
    "hello": "greeting", "thankyou": "greeting",
    "good": "description", "hot": "description", "new": "description",
}
EN = {"bank": "bank", "storeorshop": "store / shop", "black": "black", "blue": "blue", "red": "red", "white": "white",
      "shoes": "shoes", "tshirt": "T-shirt", "cellphone": "mobile phone", "pen": "pen", "biglarge": "big / large",
      "smalllittle": "small / little", "hello": "hello", "thankyou": "thank you", "good": "good", "hot": "hot", "new": "new"}
KIND_EN = {"place": "a place", "colour": "a colour", "clothing": "clothing", "object": "an object",
           "size": "a size", "greeting": "a greeting", "description": "a description (good / hot / new)"}
ALIASES = {"storeorshop": ["store", "shop"], "tshirt": ["tshirt", "t-shirt", "t shirt"], "cellphone": ["phone", "mobile", "cell phone"],
           "biglarge": ["big", "large"], "smalllittle": ["small", "little"], "thankyou": ["thanks", "thank you"],
           "hello": ["hello", "hi"], "shoes": ["shoe", "shoes"]}

# BCP-47 tag is what the browser uses for speech synthesis and recognition.
LANGS = {"hi": ("Hindi", "hi-IN"), "mr": ("Marathi", "mr-IN"), "bn": ("Bengali", "bn-IN"),
         "ta": ("Tamil", "ta-IN"), "te": ("Telugu", "te-IN")}
_ORDER = list(CONCEPTS)
_T = {
 "hi": "बैंक|दुकान|काला|नीला|लाल|सफ़ेद|जूते|टी-शर्ट|मोबाइल फ़ोन|पेन|बड़ा|छोटा|नमस्ते|धन्यवाद|अच्छा|गरम|नया",
 "mr": "बँक|दुकान|काळा|निळा|लाल|पांढरा|बूट|टी-शर्ट|मोबाइल फोन|पेन|मोठा|लहान|नमस्कार|धन्यवाद|चांगला|गरम|नवीन",
 "bn": "ব্যাংক|দোকান|কালো|নীল|লাল|সাদা|জুতো|টি-শার্ট|মোবাইল ফোন|কলম|বড়|ছোট|নমস্কার|ধন্যবাদ|ভালো|গরম|নতুন",
 "ta": "வங்கி|கடை|கருப்பு|நீலம்|சிவப்பு|வெள்ளை|காலணிகள்|டி-ஷர்ட்|கைப்பேசி|பேனா|பெரிய|சிறிய|வணக்கம்|நன்றி|நல்ல|சூடு|புதிய",
 "te": "బ్యాంకు|దుకాణం|నలుపు|నీలం|ఎరుపు|తెలుపు|బూట్లు|టీ-షర్ట్|మొబైల్ ఫోన్|పెన్ను|పెద్ద|చిన్న|నమస్కారం|ధన్యవాదాలు|మంచి|వేడి|కొత్త",
}
_ORDER_T = ["bank", "storeorshop", "black", "blue", "red", "white", "shoes", "tshirt", "cellphone", "pen",
            "biglarge", "smalllittle", "hello", "thankyou", "good", "hot", "new"]
REALIZATIONS = {lang: dict(zip(_ORDER_T, s.split("|"))) for lang, s in _T.items()}


def realize(concept, lang):
    """Concept -> text in `lang`. Raises KeyError if missing: there is NO fallback to another language."""
    return REALIZATIONS[lang][concept]


def _norm(s):
    return re.sub(r"[\s\-_.,!?]+", " ", s.lower()).strip()


def match_text(text, lang):
    """Transcript -> concept IDs in spoken order. Lexicon matching over the 17 supported concepts only."""
    t = " " + _norm(text) + " "
    hits = []
    for c in CONCEPTS:
        words = [REALIZATIONS[lang][c]] + [EN[c]] + ALIASES.get(c, []) + [a for a in EN[c].split(" / ")]
        for w in words:
            w = _norm(w)
            i = t.find(" " + w + " ") if w.isascii() else t.find(w)
            if i >= 0:
                hits.append((i, c)); break
    out = []
    for _, c in sorted(hits):
        if c not in out:
            out.append(c)
    return out
