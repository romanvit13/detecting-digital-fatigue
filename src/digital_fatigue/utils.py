import re

MENTION_RE = re.compile(r"@\w+")
RT_RE = re.compile(r"^RT\s+@\w+:\s*", flags=re.IGNORECASE)
URL_RE = re.compile(r"https?://\S+|www\.\S+")
NONWORD_RE = re.compile(r"[^a-zA-ZА-Яа-яІіЇїЄєҐґ#\s'-]")
MULTISPACE_RE = re.compile(r"\s+")

UK_STOP = {
    "і","й","та","або","але","а","у","в","на","до","з","із","зі","для","про","під","над","при","по",
    "це","цей","ця","ці","те","той","такий","таке","такі","що","щоб","як","коли","де","який","яка",
    "які","яке","не","ні","так","то","ж","же","би","б","було","бути","є","був","була","були","ми",
    "ви","вони","він","вона","воно","мене","тобі","собі","його","її","їх","наш","ваш","свій","свої",
    "дуже","ще","вже","лише","тільки","після","перед","через","тому","тут","там","ось"
}

EN_STOP = {
    "the","a","an","to","of","and","or","in","on","for","with","at","from","as","is","are","was","were",
    "it","this","that","these","those","i","you","he","she","we","they","me","my","your","our","their",
    "be","been","being","do","did","does","not","no","yes","so","just","im","i'm","dont","don't","cant","can't",
    "very","too","have","has","had","will","would","can","could","should","about","into","than","then","there",
    "here","also","if","because","while","when","where","who","whom","which","what","why","how",
    "his","her","hers","him","them","their","ours","mine","yours","itself","himself","herself",
    "one","two","three","really","still","much","many","more","most","some","any","every",
    "thing","things","stuff","someone","anyone","everyone","something","anything","everything",
    "say","says","said","told","tell","tells","make","made","let","lets","using","used","use",
    "go","going","went","come","coming","came","take","took","taken","look","looks","looked",
    "like","liked","haha","lol","yeah","okay","ok","nah","omg","amp","user","users","rt"
}

SOCIAL_MEDIA_NOISE = {
    "user","users","rt","amp","im","ive","id","ill","dont","didnt","doesnt","cant","couldnt",
    "twitter","tweet","tweets","reddit","post","posts","comment","comments",
    "thing","things","stuff","someone","anyone","everyone","anything","something",
    "guy","guys","girl","girls","boy","boys","man","men","woman","women",
    "people","person","said","says","tell","told","let","lets","using","used","one",
    "really","still","much","many","also","well","yeah","okay","ok","nah","haha","lol",
    "ive","youre","theyre","hes","shes","thats","theres","whats"
}


def stopwords_for_lang(lang="uk"):
    if lang == "uk":
        return sorted(list(UK_STOP | EN_STOP | SOCIAL_MEDIA_NOISE))
    return sorted(list(EN_STOP | SOCIAL_MEDIA_NOISE))


def normalize_for_topics(text: str) -> str:
    t = str(text)
    t = RT_RE.sub("", t)
    t = URL_RE.sub(" ", t)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("&amp;", " ")
    t = NONWORD_RE.sub(" ", t)
    t = re.sub(r"\b\d+\b", " ", t)
    t = MULTISPACE_RE.sub(" ", t).strip().lower()
    return t


def clean_keywords_string(s: str) -> str:
    s = str(s)
    s = re.sub(r"\b\d+\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" ,;")
    return s


def token_is_bad(tok: str) -> bool:
    tok = tok.strip().lower()
    if len(tok) < 3:
        return True
    if tok in SOCIAL_MEDIA_NOISE or tok in EN_STOP or tok in UK_STOP:
        return True
    if re.search(r"\d", tok):
        return True
    return False


def term_is_bad(term: str) -> bool:
    term = clean_keywords_string(term.lower())
    if not term:
        return True
    parts = [p for p in term.split() if p.strip()]
    if len(parts) == 0:
        return True
    if all(token_is_bad(p) for p in parts):
        return True
    if any(p in SOCIAL_MEDIA_NOISE or p in EN_STOP or p in UK_STOP for p in parts):
        return True
    if len(set(parts)) < len(parts):
        return True
    banned_fragments = {
        "user user", "one day", "right now", "years ago", "feel like", "look like",
        "said user", "user said", "people say", "something like"
    }
    if term in banned_fragments:
        return True
    return False


def postprocess_terms(terms, max_terms=8):
    cleaned = []
    seen = set()
    for term in terms:
        term = clean_keywords_string(term)
        if term_is_bad(term):
            continue
        parts = term.split()
        if any(token_is_bad(p) for p in parts):
            continue
        if term in seen:
            continue
        cleaned.append(term)
        seen.add(term)
        if len(cleaned) >= max_terms:
            break
    return cleaned
