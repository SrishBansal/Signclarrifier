import yaml, shutil, sys

ENTRIES_YAML = """  I:
    category: pronoun
    label:
      en: I
      hi: मैं
      ta: நான்
      bn: আমি
      te: నేను
    sign_id: i
    signed_effect: pointing
    synonyms:
      en:
      - i
      - me
      hi:
      - मैं
      - मुझे
      hinglish:
      - main
      - mai
      ta:
      - நான்
      bn:
      - আমি
      te:
      - నేను
  YOU:
    category: pronoun
    label:
      en: you
      hi: आप
      ta: நீங்கள்
      bn: আপনি
      te: మీరు
    sign_id: you
    signed_effect: pointing
    synonyms:
      en:
      - you
      hi:
      - आप
      - तुम
      hinglish:
      - aap
      - tum
      ta:
      - நீங்கள்
      bn:
      - আপনি
      te:
      - మీరు
  YOUPLURAL:
    category: pronoun
    label:
      en: you all
      hi: आप सब
      ta: நீங்கள் எல்லாரும்
      bn: আপনারা
      te: మీరందరూ
    sign_id: youplural
    signed_effect: pointing
    synonyms:
      en:
      - you all
      - you guys
      hi:
      - आप सब
      - तुम सब
      hinglish:
      - aap sab
      - tum sab
      ta:
      - நீங்கள் எல்லாரும்
      bn:
      - আপনারা
      te:
      - మీరందరూ
  HE:
    category: pronoun
    label:
      en: he
      hi: वह
      ta: அவன்
      bn: সে
      te: అతను
    sign_id: he
    signed_effect: pointing
    synonyms:
      en:
      - he
      - him
      hi:
      - वह
      hinglish:
      - woh
      ta:
      - அவன்
      bn:
      - সে
      te:
      - అతను
  SHE:
    category: pronoun
    label:
      en: she
      hi: वह
      ta: அவள்
      bn: সে
      te: ఆమె
    sign_id: she
    signed_effect: pointing
    synonyms:
      en:
      - she
      - her
      hi:
      - वह
      hinglish:
      - woh
      ta:
      - அவள்
      bn:
      - সে
      te:
      - ఆమె
  IT:
    category: pronoun
    label:
      en: it
      hi: यह
      ta: இது
      bn: এটি
      te: ఇది
    sign_id: it
    signed_effect: pointing
    synonyms:
      en:
      - it
      hi:
      - यह
      hinglish:
      - yeh
      ta:
      - இது
      bn:
      - এটি
      te:
      - ఇది
  WE:
    category: pronoun
    label:
      en: we
      hi: हम
      ta: நாங்கள்
      bn: আমরা
      te: మేము
    sign_id: we
    signed_effect: pointing
    synonyms:
      en:
      - we
      - us
      hi:
      - हम
      hinglish:
      - hum
      ta:
      - நாங்கள்
      bn:
      - আমরা
      te:
      - మేము
  THEY:
    category: pronoun
    label:
      en: they
      hi: वे
      ta: அவர்கள்
      bn: তারা
      te: వారు
    sign_id: they
    signed_effect: pointing
    synonyms:
      en:
      - they
      - them
      hi:
      - वे
      hinglish:
      - ve
      ta:
      - அவர்கள்
      bn:
      - তারা
      te:
      - వారు
  LIGHT:
    category: descriptor
    label:
      en: light
      hi: हल्का
      ta: இலேசான
      bn: হালকা
      te: తేలికైన
    sign_id: light
    signed_effect: attribute
    synonyms:
      en:
      - light
      - lightweight
      hi:
      - हल्का
      hinglish:
      - halka
      ta:
      - இலேசான
      bn:
      - হালকা
      te:
      - తేలికైన
  MEAN:
    category: descriptor
    label:
      en: mean (signify)
      hi: मतलब
      ta: அர்த்தம்
      bn: অর্থ
      te: అర్థం
    sign_id: mean
    signed_effect: attribute
    synonyms:
      en:
      - mean
      - meaning
      hi:
      - मतलब
      hinglish:
      - matlab
      ta:
      - அர்த்தம்
      bn:
      - অর্থ
      te:
      - అర్థం
  SECOND:
    category: time_unit
    label:
      en: second
      hi: सेकंड
      ta: நொடி
      bn: সেকেন্ড
      te: సెకను
    sign_id: second
    signed_effect: attribute
    synonyms:
      en:
      - second
      - seconds
      hi:
      - सेकंड
      hinglish:
      - second
      ta:
      - நொடி
      bn:
      - সেকেন্ড
      te:
      - సెకను
  TIME:
    category: time_unit
    label:
      en: time
      hi: समय
      ta: நேரம்
      bn: সময়
      te: సమయం
    sign_id: time
    signed_effect: attribute
    synonyms:
      en:
      - time
      hi:
      - समय
      hinglish:
      - samay
      ta:
      - நேரம்
      bn:
      - সময়
      te:
      - సమయం
"""

p = "data/ontology.yaml"
shutil.copy(p, p + ".bak5")

with open(p, "a", encoding="utf-8") as f:
    f.write("\n" + ENTRIES_YAML)

# Validate with the real loader, not just yaml.safe_load, so the include_classes
# check and the multi-language requirement both get exercised before we trust it.
sys.path.insert(0, ".")
from core.ontology import Ontology
o = Ontology(p)
print(f"Loads cleanly: {len(o.concepts)} concepts")
for w in ["I", "YOU", "YOUPLURAL", "HE", "SHE", "IT", "WE", "THEY", "LIGHT", "MEAN", "SECOND", "TIME"]:
    c = o.concepts[w]
    print(w, "-> sign_id:", c["sign_id"], "| has hi:", "hi" in c["label"])
