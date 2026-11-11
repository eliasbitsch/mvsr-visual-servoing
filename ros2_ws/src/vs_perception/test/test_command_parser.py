from vs_perception.command_parser import (GOTO, STOP, parse_command, parse_reply, parse_rules,
                                          rules_confident)


def test_rules_german_goto():
    assert parse_rules('fahre über die rote Dose') == (GOTO, 'red can')
    assert parse_rules('Bitte fahr über den gelben Ball') == (GOTO, 'yellow ball')
    assert parse_rules('über die blaue Schachtel') == (GOTO, 'blue box')


def test_rules_english_goto():
    assert parse_rules('move above the blue box') == (GOTO, 'blue box')
    assert parse_rules('red can') == (GOTO, 'red can')


def test_rules_stop():
    assert parse_rules('stop') == (STOP, '')
    assert parse_rules('Stopp!') == (STOP, '')
    assert parse_rules('bitte halt') == (STOP, '')


def test_rules_empty():
    assert parse_rules('') is None
    assert parse_rules('fahre über die') is None


def test_reply_json_with_prose_and_thinking():
    text = '<think>the user wants the can</think>Sure: {"action": "goto", "object": "Red can."}'
    assert parse_reply(text) == (GOTO, 'red can')
    assert parse_reply('```json\n{"action": "stop"}\n```') == (STOP, '')


def test_reply_unusable():
    assert parse_reply('I cannot do that') is None
    assert parse_reply('{"action": "goto"}') is None
    assert parse_reply('{"action": "dance", "object": "red can"}') is None


def test_rules_confident_only_for_known_words():
    assert rules_confident((STOP, ''))
    assert rules_confident(parse_rules('fahre über den grünen Becher'))
    assert not rules_confident(parse_rules('fahre zum Ding neben dem Ball'))
    assert not rules_confident(None)


def test_simple_command_skips_the_model():
    # unreachable server: a known command must not even try it
    assert parse_command('fahre über die rote Dose', 'http://127.0.0.1:9/v1', 'x', 1.0) \
        == (GOTO, 'red can', 'rules')


def test_unknown_words_fall_back_to_rules_when_server_down():
    assert parse_command('fahre zur Zahnbürste', 'http://127.0.0.1:9/v1', 'x', 1.0) \
        == (GOTO, 'zahnbürste', 'rules')
