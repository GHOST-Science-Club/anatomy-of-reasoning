import json
from pathlib import Path

import pytest

from labeling.splitting import process_response_to_sentences as split


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('One. Two? Three!', ['One.', 'Two?', 'Three!']),
        ('First\n\nSecond', ['First', 'Second']),
        ('Use $a. b? c!$ now. End.', ['Use $a. b? c!$ now.', 'End.']),
        ('Use $$a. b? c!$$ now. End.', ['Use $$a. b? c!$$ now.', 'End.']),
        ('Dr. Smith uses e.g. apples. Done.', ['Dr. Smith uses e.g. apples.', 'Done.']),
        ('Thus:\n\nx is one.', ['Thus: x is one.']),
        ('x\n\n= 2.', ['x = 2.']),  # ThinkARM merges leading equals, not trailing equals.
        ('<think>Think.\n\nThus:</think>Answer.', ['Think.', 'Thus:', 'Answer.']),
        ('', []),
        (' \n\n ', []),
    ],
)
def test_small_cases(text, expected):
    assert split(text) == expected


def test_all_human_prefixes():
    root = Path('ThinkARM/data')
    files = sorted((root / 'ground_truth').glob('*/*.json'))
    if not files or not (root / 'raw').exists():
        pytest.skip('ThinkARM submodule human/raw data is missing')
    for path in files:
        raw = json.loads((root / 'raw' / f'{path.parent.name}.json').read_text())
        human = json.loads(path.read_text())
        assert split(raw[int(path.stem) - 1]['Response'])[: len(human)] == [
            row['sentence'] for row in human
        ], str(path)
