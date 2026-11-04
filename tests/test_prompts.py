from pathlib import Path

import pytest

from labeling.labelers import OpenAICompatLabeler, build_prompt


def test_prompt_content_order_and_braces():
    prompt = build_prompt(
        'Problem {n} {sentences} {x}', '', ['a', 'b'], guidebook='Guide {problem}'
    )
    tags = ['[Guidebook]\n', '[Math Problem]\n', '[Previous Context]\n', '[Input]\n', '[Format]\n']
    positions = [prompt.index(tag) for tag in tags]
    assert positions == sorted(positions)
    for text in [
        'Guide {problem}',
        'Problem {n} {sentences} {x}',
        '[1] a\n[2] b',
        'There is no previous sentences.',
        'label for 2 sentences.',
    ]:
        assert text in prompt
    assert '{guidebook}' not in prompt
    assert '{context}' not in prompt
    assert 'corresponding labels' not in Path('src/labeling/prompts/instructions.md').read_text()


def test_context_rendered():
    prompt = build_prompt('P', 'earlier\nsentences', ['a'], guidebook='G')
    assert 'earlier\nsentences' in prompt
    assert 'label for 1 sentences.' in prompt


def test_missing_guidebook_fails_at_startup(tmp_path):
    with pytest.raises(FileNotFoundError, match='guide'):
        OpenAICompatLabeler({'model': 'M', 'guidebook': str(tmp_path / 'missing-guide.md')})
