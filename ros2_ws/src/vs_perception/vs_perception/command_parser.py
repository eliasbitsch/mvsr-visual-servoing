"""Turns a typed command ("fahre über die rote Dose", "move above the yellow ball", "stop")
into an action and an English object prompt for the detector.

The language model (same OpenAI-compatible server as the VLM detector) does the translation;
a small rule set is the fallback when the server is unreachable or answers nonsense.
"""
import json
import re
import urllib.request

STOP = 'stop'
GOTO = 'goto'

_STOP_WORDS = {'stop', 'stopp', 'halt', 'anhalten', 'abbrechen', 'cancel', 'abort'}

# Filler words removed by the rule-based fallback (German and English)
_FILLER = re.compile(
    r'\b(bitte|please|fahre?|fahr|bewege?|beweg|geh|gehe|go|move|drive|navigate|servo|positioniere?|'
    r'dich|den|die|das|dem|der|einen|eine|ein|the|a|an|to|über|ueber|uber|oberhalb|zu|zum|zur|'
    r'above|over|on top of|onto|roboter|robot|kamera|camera|nach|hin)\b', re.I)

# Small German -> English dictionary for the fallback, enough for the lab objects
_DE_EN = {
    'rote': 'red', 'roten': 'red', 'roter': 'red', 'rot': 'red',
    'blaue': 'blue', 'blauen': 'blue', 'blauer': 'blue', 'blau': 'blue',
    'gelbe': 'yellow', 'gelben': 'yellow', 'gelber': 'yellow', 'gelb': 'yellow',
    'grüne': 'green', 'grünen': 'green', 'grüner': 'green', 'grün': 'green',
    'weiße': 'white', 'weißen': 'white', 'schwarze': 'black', 'schwarzen': 'black',
    'dose': 'can', 'schachtel': 'box', 'box': 'box', 'kiste': 'box', 'karton': 'box',
    'becher': 'cup', 'tasse': 'cup', 'ball': 'ball', 'kugel': 'ball',
    'marker': 'marker', 'tafel': 'board',
}

# Words the rules may output without asking the model: a rule result made only of these is
# trusted, anything else ("the thing next to the ball") goes to the language model.
_KNOWN = set(_DE_EN.values()) | {'mug', 'aruco', 'board', 'orange', 'purple', 'grey', 'gray',
                                 'small', 'big', 'large'}

SYSTEM_PROMPT = (
    'You turn short robot commands (German or English) into JSON for a visual servoing robot. '
    'The robot can only move its camera above one object or stop. '
    'Answer with JSON only, no prose: {"action": "goto", "object": "<object>"} or '
    '{"action": "stop"}. <object> is a short English noun phrase an open-vocabulary object '
    'detector understands, with colour if given, without articles, e.g. "red can", '
    '"blue box", "yellow ball".')


def _strip_thinking(text):
    return re.sub(r'<think>.*?</think>', '', text, flags=re.S).strip()


def parse_reply(text):
    """Parse the model answer into (action, object) or None if it is unusable."""
    match = re.search(r'\{.*?\}', _strip_thinking(text), re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    action = str(data.get('action', '')).lower()
    if action == STOP:
        return STOP, ''
    obj = str(data.get('object', '')).strip().strip('."\'').lower()
    if action == GOTO and obj and len(obj.split()) <= 5:
        return GOTO, obj
    return None


def parse_rules(command):
    """Rule-based fallback: stop words, then strip filler and translate German words."""
    words = re.findall(r'[\wäöüß]+', command.lower())
    if not words:
        return None
    if words[0] in _STOP_WORDS or (len(words) <= 2 and _STOP_WORDS & set(words)):
        return STOP, ''
    rest = _FILLER.sub(' ', command.lower())
    obj = [_DE_EN.get(w, w) for w in re.findall(r'[\wäöüß]+', rest)]
    return (GOTO, ' '.join(obj)) if obj else None


def ask_model(command, url, model, timeout=30.0):
    body = {
        'model': model,
        'temperature': 0,
        'max_tokens': 256,
        'messages': [{'role': 'system', 'content': SYSTEM_PROMPT},
                     {'role': 'user', 'content': command}],
    }
    req = urllib.request.Request(f'{url}/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)['choices'][0]['message']['content']


def rules_confident(result):
    """True if the rule result needs no model: a stop, or an object of known words only."""
    if result is None:
        return False
    action, obj = result
    return action == STOP or all(w in _KNOWN for w in obj.split())


def parse_command(command, url=None, model=None, timeout=30.0):
    """Return (action, object, source) with source 'model' or 'rules', or None.

    Simple commands ("fahre über die rote Dose", "stopp") are resolved by the rules at once;
    the model is only asked for the rest, because it shares the server with the detector and
    has to wait for the running image request.
    """
    ruled = parse_rules(command)
    if rules_confident(ruled):
        return (*ruled, 'rules')
    if url and model:
        try:
            result = parse_reply(ask_model(command, url, model, timeout))
            if result:
                return (*result, 'model')
        except OSError:
            pass  # server down or timeout -> rules
    return (*ruled, 'rules') if ruled else None
