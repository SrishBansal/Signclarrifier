#!/usr/bin/env python3
"""Build data/ontology_candidate.yaml: current ontology + concepts for new INCLUDE signs.
Run from project root.   python3 scripts/build_candidate_ontology.py
Test lookups:            python3 scripts/build_candidate_ontology.py --test "good morning" tall
Translations are machine-authored drafts: native speakers must review them."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0, ".")
from core.ontology import Ontology

ONT = Path("data/ontology.yaml")
OUT = Path("data/ontology_candidate.yaml")
INC = Path("data/include_classes.txt")
CAND = Path("models/signs_include_candidate.json")

PRON = "pronoun: appears in nearly every sentence; decide how pointing signs should be handled"
HOLD = {k: PRON for k in ["i", "you", "youplural", "he", "she", "it", "we", "they"]}
HOLD.update({
    "light": "ambiguous (lamp / not heavy / pale colour)",
    "mean": "ambiguous (verb / unkind / average)",
    "second": "ambiguous (ordinal vs time unit)",
    "time": "ambiguous (clock time vs occasion); 'what time' needs phrase handling",
})
# phrases that currently sit under HELLO but now have their own signs
TAKEOVER = {"good morning", "good evening", "good afternoon"}

# key, category, signed_effect, en, hi, ta, bn, te, extra_en_synonyms, hinglish_synonyms
R = [
("afternoon","time","attribute","afternoon","दोपहर","மதியம்","দুপুর","మధ్యాహ్నం",[],["dopahar"]),
("alive","descriptor","attribute","alive","जीवित","உயிருடன்","জীবিত","సజీవ",["living"],["zinda","jeevit"]),
("alright","descriptor","attribute","alright","ठीक है","சரி","ঠিক আছে","సరే",["all right"],["theek hai","thik hai"]),
("bad","attribute_state","attribute","bad","बुरा","மோசம்","খারাপ","చెడు",[],["kharab","bura"]),
("beautiful","attribute_state","attribute","beautiful","सुंदर","அழகான","সুন্দর","అందమైన",["pretty"],["sundar","khoobsurat"]),
("blind","descriptor","attribute","blind","अंधा","பார்வையற்ற","অন্ধ","గుడ్డి",[],["andha"]),
("clean","attribute_state","attribute","clean","साफ़","சுத்தமான","পরিষ্কার","శుభ్రమైన",[],["saaf","saf"]),
("clothing","product","product","clothing","कपड़े","உடை","পোশাক","దుస్తులు",["clothes","garment","garments"],["kapde","kapda"]),
("colour","product","product","colour","रंग","நிறம்","রং","రంగు",["color","colours","colors"],["rang"]),
("cool","attribute_state","attribute","cool","शीतल","குளிர்ச்சியான","শীতল","చల్లని",[],[]),
("curved","attribute_state","attribute","curved","मुड़ा हुआ","வளைந்த","বাঁকা","వంకర",["curve","bent"],["teda"]),
("dead","descriptor","attribute","dead","मृत","இறந்த","মৃত","చనిపోయిన",[],["mrit"]),
("deaf","descriptor","attribute","deaf","बहरा","காது கேளாத","বধির","చెవిటి",[],["behra"]),
("deep","attribute_size","attribute","deep","गहरा","ஆழமான","গভীর","లోతైన",[],["gehra","gahra"]),
("dirty","attribute_state","attribute","dirty","गंदा","அழுக்கான","নোংরা","మురికి",[],["ganda","gandi"]),
("dry","attribute_state","attribute","dry","सूखा","வறண்ட","শুকনো","పొడి",[],["sukha","sookha"]),
("evening","time","attribute","evening","शाम","மாலை","সন্ধ্যা","సాయంత్రం",[],["shaam"]),
("extra","attribute_state","attribute","extra","अतिरिक्त","கூடுதல்","অতিরিক্ত","అదనపు",["additional"],["extra","zyada"]),
("famous","descriptor","attribute","famous","प्रसिद्ध","பிரபலமான","বিখ্যাত","ప్రసిద్ధ",["popular"],["mashhoor","prasiddh"]),
("fast","attribute_state","attribute","fast","तेज़","வேகமான","দ্রুত","వేగంగా",["quick","quickly"],["tez","jaldi"]),
("female","descriptor","attribute","female","महिला","பெண்","মহিলা","స్త్రీ",["woman","women","lady","ladies"],["mahila","aurat"]),
("flat","attribute_state","attribute","flat","सपाट","தட்டையான","সমতল","చదునైన",[],["sapaat","chapta"]),
("friday","time","attribute","friday","शुक्रवार","வெள்ளிக்கிழமை","শুক্রবার","శుక్రవారం",[],["shukravar"]),
("goodafternoon","social","greeting","good afternoon","शुभ दोपहर","இனிய மதிய வணக்கம்","শুভ অপরাহ্ন","శుభ మధ్యాహ్నం",[],["shubh dopahar"]),
("goodevening","social","greeting","good evening","शुभ संध्या","இனிய மாலை வணக்கம்","শুভ সন্ধ্যা","శుభ సాయంత్రం",[],["shubh sandhya"]),
("goodmorning","social","greeting","good morning","सुप्रभात","காலை வணக்கம்","সুপ্রভাত","శుభోదయం",[],["suprabhat"]),
("goodnight","social","greeting","good night","शुभ रात्रि","இனிய இரவு","শুভ রাত্রি","శుభ రాత్రి",["goodnight"],["shubh ratri"]),
("howareyou","social","greeting","how are you","आप कैसे हैं","நீங்கள் எப்படி இருக்கிறீர்கள்","আপনি কেমন আছেন","మీరు ఎలా ఉన్నారు",["how do you do"],["aap kaise hain","kaise ho"]),
("happy","descriptor","attribute","happy","खुश","மகிழ்ச்சி","খুশি","సంతోషం",["glad","cheerful"],["khush"]),
("hard","attribute_state","attribute","hard","कठोर","கடினமான","শক্ত","కఠినమైన",["firm"],["kadak","sakht"]),
("healthy","descriptor","attribute","healthy","स्वस्थ","ஆரோக்கியமான","সুস্থ","ఆరోగ్యకరమైన",[],["swasth","tandurust"]),
("heavy","attribute_state","attribute","heavy","भारी","கனமான","ভারী","బరువైన",[],["bhaari","bhari"]),
("high","attribute_size","attribute","high","ऊँचा","உயர்ந்த","উঁচু","ఎత్తైన",[],["ooncha","uncha"]),
("hour","time","attribute","hour","घंटा","மணி நேரம்","ঘণ্টা","గంట",["hours"],["ghanta","ghante"]),
("long","attribute_size","attribute","long","लंबा","நீளமான","লম্বা","పొడవైన",[],["lamba","lambi"]),
("loose","attribute_size","attribute","loose","ढीला","தளர்வான","ঢিলা","వదులు",[],["dhila","dheela"]),
("loud","attribute_state","attribute","loud","तेज़ आवाज़","சத்தமான","জোরে","పెద్ద శబ్దం",["noisy"],["shor","zor se"]),
("low","attribute_size","attribute","low","नीचा","தாழ்வான","নিচু","తక్కువ",[],["neecha","nicha"]),
("male","descriptor","attribute","male","पुरुष","ஆண்","পুরুষ","పురుషుడు",["man","men","gents"],["purush","aadmi"]),
("minute","time","attribute","minute","मिनट","நிமிடம்","মিনিট","నిమిషం",["minutes"],["minute"]),
("monday","time","attribute","monday","सोमवार","திங்கட்கிழமை","সোমবার","సోమవారం",[],["somvar"]),
("month","time","attribute","month","महीना","மாதம்","মাস","నెల",["months"],["mahina","mahine"]),
("morning","time","attribute","morning","सुबह","காலை","সকাল","ఉదయం",[],["subah"]),
("narrow","attribute_size","attribute","narrow","संकरा","குறுகிய","সরু","ఇరుకైన",[],["sankra"]),
("nice","attribute_state","attribute","nice","बढ़िया","அருமையான","চমৎকার","చక్కని",["lovely"],["badhiya","mast"]),
("night","time","attribute","night","रात","இரவு","রাত","రాత్రి",[],["raat"]),
("old","attribute_state","attribute","old","पुराना","பழைய","পুরানো","పాత",[],["purana","purani"]),
("pleased","descriptor","attribute","pleased","प्रसन्न","மகிழ்ச்சியடைந்த","সন্তুষ্ট","సంతృప్తి",["satisfied"],["prasann","santusht"]),
("pocket","product","product","pocket","जेब","பாக்கெட்","পকেট","జేబు",["pockets"],["jeb"]),
("poor","descriptor","attribute","poor","गरीब","ஏழை","গরিব","పేద",[],["garib","gareeb"]),
("quiet","attribute_state","attribute","quiet","शांत","அமைதியான","শান্ত","నిశ్శబ్దం",["silent"],["shant","chup"]),
("rich","descriptor","attribute","rich","अमीर","பணக்கார","ধনী","ధనవంతుడు",["wealthy"],["amir","ameer"]),
("sad","descriptor","attribute","sad","दुखी","சோகமான","দুঃখী","విచారం",["unhappy"],["dukhi","udaas"]),
("saturday","time","attribute","saturday","शनिवार","சனிக்கிழமை","শনিবার","శనివారం",[],["shanivar"]),
("shallow","attribute_size","attribute","shallow","उथला","ஆழமற்ற","অগভীর","లోతులేని",[],["uthla"]),
("short","attribute_size","attribute","short","कम लंबा","குட்டையான","খাটো","పొట్టి",[],["kam lamba"]),
("sick","descriptor","attribute","sick","बीमार","நோயுற்ற","অসুস্থ","అనారోగ్యం",["ill","unwell"],["bimar","beemar"]),
("slow","attribute_state","attribute","slow","धीमा","மெதுவான","ধীর","నెమ్మదిగా",["slowly"],["dheere","dheema"]),
("soft","attribute_state","attribute","soft","मुलायम","மென்மையான","নরম","మెత్తని",[],["mulayam","naram"]),
("strong","attribute_state","attribute","strong","मज़बूत","வலுவான","শক্তিশালী","బలమైన",[],["mazboot","majboot"]),
("sunday","time","attribute","sunday","रविवार","ஞாயிற்றுக்கிழமை","রবিবার","ఆదివారం",[],["ravivar","itwaar"]),
("tall","attribute_size","attribute","tall","लंबा कद","உயரமான","লম্বা","ఎత్తు",[],["lamba kad"]),
("thick","attribute_size","attribute","thick","मोटा","தடிமனான","মোটা","మందపాటి",[],["mota","moti"]),
("thin","attribute_size","attribute","thin","पतला","மெல்லிய","পাতলা","సన్నని",["slim"],["patla","patli"]),
("thursday","time","attribute","thursday","गुरुवार","வியாழக்கிழமை","বৃহস্পতিবার","గురువారం",[],["guruvar","brihaspativar"]),
("tight","attribute_size","attribute","tight","तंग","இறுக்கமான","আঁটসাঁট","బిగుతైన",[],["tang","kasa hua"]),
("today","time","attribute","today","आज","இன்று","আজ","ఈరోజు",[],["aaj"]),
("tomorrow","time","attribute","tomorrow","आने वाला कल","நாளை","আগামীকাল","రేపు",[],[]),
("tuesday","time","attribute","tuesday","मंगलवार","செவ்வாய்க்கிழமை","মঙ্গলবার","మంగళవారం",[],["mangalvar"]),
("ugly","attribute_state","attribute","ugly","बदसूरत","அசிங்கமான","কুৎসিত","అసహ్యకరమైన",[],["badsurat"]),
("warm","attribute_state","attribute","warm","गुनगुना","வெதுவெதுப்பான","উষ্ণ","గోరువెచ్చని",[],["gunguna"]),
("weak","attribute_state","attribute","weak","कमज़ोर","பலவீனமான","দুর্বল","బలహీనమైన",[],["kamzor","kamjor"]),
("wednesday","time","attribute","wednesday","बुधवार","புதன்கிழமை","বুধবার","బుధవారం",[],["budhvar"]),
("week","time","attribute","week","सप्ताह","வாரம்","সপ্তাহ","వారం",["weeks"],["hafta","saptah","hafte"]),
("wet","attribute_state","attribute","wet","गीला","ஈரமான","ভেজা","తడి",[],["geela","gila"]),
("wide","attribute_size","attribute","wide","चौड़ा","அகலமான","চওড়া","వెడల్పైన",["broad"],["chauda"]),
("year","time","attribute","year","साल","வருடம்","বছর","సంవత్సరం",["years"],["saal","varsh"]),
("yesterday","time","attribute","yesterday","बीता हुआ कल","நேற்று","গতকাল","నిన్న",[],[]),
("young","descriptor","attribute","young","जवान","இளம்","তরুণ","యువ",[],["jawan","yuva"]),
]
ROWS = {r[0]: r for r in R}
norm = Ontology.normalize_text

def frame_stats(seq):
    a = np.asarray(seq, dtype=float)
    if a.ndim != 2 or a.shape[1] != 225:
        return None
    zero = int((np.abs(a).sum(axis=1) == 0).sum())
    hands = int((np.abs(a[:, 99:]).sum(axis=1) > 0).sum())  # assumes pose first (99), then hands
    return a.shape[0], zero, hands

def build():
    raw = yaml.safe_load(ONT.read_text(encoding="utf-8"))
    concepts = raw["concepts"]
    cand = json.loads(CAND.read_text())
    inc = {l.strip().lower() for l in INC.read_text().splitlines() if l.strip()}
    used = {str(v["sign_id"]).lower() for v in concepts.values() if v.get("sign_id")}
    new_keys = [k for k in cand if k.lower() not in used]

    # 1. HELLO takeover
    removed = []
    for cid, d in concepts.items():
        for lang, lst in (d.get("synonyms") or {}).items():
            keep = []
            for s in lst or []:
                if norm(str(s)) in TAKEOVER:
                    removed.append((cid, lang, s))
                else:
                    keep.append(s)
            d["synonyms"][lang] = keep

    # 2. index of existing synonyms
    idx = {}
    for cid, d in concepts.items():
        idx[cid.lower()] = cid
        for lst in (d.get("synonyms") or {}).values():
            for s in lst or []:
                n = norm(str(s))
                if n:
                    idx[n] = cid

    held, no_row, no_inc, dropped, added = [], [], [], [], []
    for k in new_keys:
        if k in HOLD:
            held.append((k, HOLD[k])); continue
        if k not in ROWS:
            no_row.append(k); continue
        if k.lower() not in inc:
            no_inc.append(k); continue
        _, cat, eff, en, hi, ta, bn, te, en_syn, hing = ROWS[k]
        cid = k.upper()
        if cid in concepts:
            dropped.append((cid, "<concept id>", "already exists")); continue
        idx[cid.lower()] = cid
        syn = {"bn": [bn], "en": [en] + en_syn, "hi": [hi], "hinglish": hing, "ta": [ta], "te": [te]}
        clean = {}
        for lang, lst in syn.items():
            out = []
            for s in lst:
                n = norm(s)
                if not n or n == cid.lower():
                    continue
                if n in idx and idx[n] != cid:
                    dropped.append((cid, s, f"already owned by {idx[n]}")); continue
                idx[n] = cid
                out.append(s)
            clean[lang] = out
        concepts[cid] = {"category": cat,
                         "label": {"bn": bn, "en": en, "hi": hi, "ta": ta, "te": te},
                         "sign_id": k, "signed_effect": eff, "synonyms": clean}
        added.append(k)

    OUT.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    ont = Ontology(str(OUT))  # runs the loader's own validation
    print(f"wrote {OUT}: {len(ont.concepts)} concepts "
          f"({sum(1 for c in ont.concepts if ont.get_sign_id(c))} with signs)")
    print(f"\nADDED {len(added)} concepts")
    print(f"\nHELD for your decision ({len(held)}):")
    for k, why in held: print(f"  {k}: {why}")
    print(f"\nNEW KEYS WITH NO TABLE ROW ({len(no_row)}): {no_row}")
    print(f"NEW KEYS NOT IN include_classes.txt ({len(no_inc)}): {no_inc}")
    print(f"\nREMOVED FROM EXISTING CONCEPTS ({len(removed)}):")
    for r in removed: print("  ", r)
    print(f"\nSYNONYMS DROPPED DUE TO COLLISIONS ({len(dropped)}):")
    for r in dropped: print("  ", r)
    print("\nSIGN DATA CHECK (new signs only; frames, all-zero frames, frames with hand data):")
    weak = []
    for k in added:
        st = frame_stats(cand[k])
        if st is None:
            print(f"  {k}: unexpected shape"); continue
        if st[2] < 12:
            weak.append((k, st))
    print(f"  weak (fewer than 12 frames with hand data): {len(weak)}")
    for k, st in weak: print(f"    {k}: frames={st[0]} zero={st[1]} hand_frames={st[2]}")

DEFAULT_TESTS = ["expensive", "price", "hello", "good morning", "good night", "how are you",
                 "clothing", "tomorrow", "tall", "happy", "xylophone"]

def run_tests(words):
    ont = Ontology(str(OUT))
    signs = json.loads(CAND.read_text())
    for w in words or DEFAULT_TESTS:
        cid = ont.resolve_concept(w)
        sid = ont.get_sign_id(cid) if cid else None
        print(f"{w!r:18} -> concept={cid} category={ont.get_category(cid) if cid else None} "
              f"sign_id={sid} in_signs={sid in signs if sid else False}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", nargs="*")
    a = ap.parse_args()
    run_tests(a.test) if a.test is not None else build()
