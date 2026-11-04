"""Split reasoning traces into sentences, ported from the ThinkARM pipeline.

The splitter is deliberately heuristic: it avoids breaking inside LaTeX math,
after common abbreviations, inside decimal numbers and inside math ellipses.
"""

ABBREVIATIONS = frozenset(
    {
        'e.g.',
        'i.e.',
        'v.s.',
        'cf.',
        'et al.',
        'ibid.',
        'etc.',
        'vs.',
        'viz.',
        'Dr.',
        'Mr.',
        'Mrs.',
        'Ms.',
        'Prof.',
        'Rev.',
        'St.',
        'Jr.',
        'Sr.',
        'Inc.',
        'Ltd.',
        'Corp.',
        'Co.',
        'LLC.',
        'Ph.D.',
        'M.D.',
        'B.A.',
        'M.A.',
        'U.S.',
        'U.K.',
        'U.S.A.',
        'N.Y.',
        'L.A.',
        'D.C.',
        'a.m.',
        'p.m.',
        'No.',
        'Vol.',
        'pp.',
        'Fig.',
        'Eq.',
        'Ref.',
        'Sec.',
        'Ch.',
        'App.',
    }
)
MATH_INDICATORS = (
    '+',
    '-',
    '*',
    '/',
    '=',
    '(',
    ')',
    '[',
    ']',
    'g(',
    'f(',
    'h(',
    'times',
    'integer',
    'induction',
)


def split_response_into_paragraphs(response):
    return [paragraph.strip() for paragraph in response.split('\n\n')]


def is_abbreviation_context(text, position):
    for abbreviation in ABBREVIATIONS:
        length = len(abbreviation)
        for offset in range(length):
            start = position - offset
            end = start + length
            if start < 0 or end > len(text):
                continue
            if text[start:end].lower() == abbreviation.lower() and (
                start == 0 or not text[start - 1].isalnum()
            ):
                return True
    return False


def is_math_ellipsis(before, after):
    before, after = before.lower(), after.lower()
    return any(token in before or token in after for token in MATH_INDICATORS)


def split_paragraph_into_sentences(paragraph):
    """Split a paragraph at sentence-ending punctuation, outside math spans."""
    splits = []
    current = ''
    i = 0
    math_delimiter = None  # '$' or '$$' while inside a math span
    while i < len(paragraph):
        char = paragraph[i]
        current += char
        next_is_dollar = i + 1 < len(paragraph) and paragraph[i + 1] == '$'
        if char == '$':
            if math_delimiter is None:
                if next_is_dollar:
                    math_delimiter = '$$'
                    current += '$'
                    i += 1
                else:
                    math_delimiter = '$'
            elif math_delimiter == '$$' and next_is_dollar:
                current += '$'
                i += 1
                math_delimiter = None
            elif math_delimiter == '$':
                math_delimiter = None
        elif math_delimiter is None:
            if char == '.' and paragraph[i + 1 : i + 3] == '..' and i + 2 < len(paragraph):
                current += paragraph[i + 1 : i + 3]
                i += 2
                before = current[-20:]
                after = paragraph[i + 1 : i + 21]
                if not is_math_ellipsis(before, after):
                    splits.append(current.strip())
                    current = ''
            elif char in '.?!':
                if char == '.' and is_abbreviation_context(paragraph, i):
                    pass
                elif char == '.' and 0 < i < len(paragraph) - 1:
                    previous, following = paragraph[i - 1], paragraph[i + 1]
                    is_decimal = previous.isdigit() and following.isdigit()
                    is_list_marker = i == 1 and previous.isdigit()
                    if not (is_decimal or is_list_marker):
                        splits.append(current.strip())
                        current = ''
                else:
                    splits.append(current.strip())
                    current = ''
        i += 1
    if current:
        splits.append(current.strip())
    return splits


def is_valid_sentence(sentence):
    """Reject empty, dash-only and punctuation-only fragments."""
    cleaned = sentence.strip()
    if not cleaned or set(cleaned) == {'-'}:
        return False
    return any(char.isalnum() for char in cleaned)


def process_section(section_text, section_type):
    return [
        {'sentence': sentence, 'type': section_type}
        for paragraph in split_response_into_paragraphs(section_text)
        for sentence in split_paragraph_into_sentences(paragraph)
        if is_valid_sentence(sentence)
    ]


def process_response_to_sentences(response, apply_merging=True):
    """Return the sentences of a response, thinking part first, then the answer.

    Text before `</think>` is the thinking part; without the tag the whole
    response counts as the answer.
    """
    if '</think>' in response:
        thinking, answer = response.split('</think>', 1)
        sentences = process_section(thinking.strip(), 'think') + process_section(
            answer.strip(), 'answer'
        )
    else:
        sentences = process_section(response, 'answer')
    if apply_merging:
        sentences = merge_colon_and_equals_sentences(sentences)
    return [item['sentence'] for item in sentences]


def _clean(item):
    """Copy a sentence record with any `<think>` marker removed."""
    item = item.copy()
    item['sentence'] = item['sentence'].replace('<think>', '').strip()
    return item


def merge_colon_and_equals_sentences(sentences):
    """Join sentences ending in ':' with the next, and '='-leading ones with the previous.

    Merging only happens between sentences of the same type.
    """
    merged = []
    i = 0
    while i < len(sentences):
        current = _clean(sentences[i])
        while current['sentence'].rstrip().endswith(':') and i + 1 < len(sentences):
            following = _clean(sentences[i + 1])
            if current['type'] != following['type']:
                break
            current['sentence'] += ' ' + following['sentence']
            i += 1
        merged.append(current)
        i += 1
    final = []
    for item in map(_clean, merged):
        if (
            item['sentence'].lstrip().startswith('=')
            and final
            and final[-1]['type'] == item['type']
        ):
            final[-1]['sentence'] += ' ' + item['sentence']
        else:
            final.append(item)
    return final
