def split_response_into_paragraphs(response):
    return [p.strip() for p in response.split('\n\n')]

def split_paragraph_into_sentences(paragraph):
    splits = []
    current = ''
    i = 0
    in_math_block = False
    math_delimiter = None
    abbreviations = {'e.g.', 'i.e.', 'v.s.', 'cf.', 'et al.', 'ibid.', 'etc.', 'vs.', 'viz.', 'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'Rev.', 'St.', 'Jr.', 'Sr.', 'Inc.', 'Ltd.', 'Corp.', 'Co.', 'LLC.', 'Ph.D.', 'M.D.', 'B.A.', 'M.A.', 'U.S.', 'U.K.', 'U.S.A.', 'N.Y.', 'L.A.', 'D.C.', 'a.m.', 'p.m.', 'No.', 'Vol.', 'pp.', 'Fig.', 'Eq.', 'Ref.', 'Sec.', 'Ch.', 'App.'}

    def is_abbreviation_context(text, position):
        for abbrev in abbreviations:
            abbrev_len = len(abbrev)
            start_pos = position - abbrev_len + 1
            if start_pos >= 0 and position + 1 <= len(text):
                potential_abbrev = text[start_pos:position + 1]
                if potential_abbrev.lower() == abbrev.lower():
                    if start_pos == 0 or not text[start_pos - 1].isalnum():
                        return True
        for abbrev in abbreviations:
            abbrev_len = len(abbrev)
            for start_offset in range(abbrev_len):
                start_pos = position - start_offset
                end_pos = start_pos + abbrev_len
                if start_pos >= 0 and end_pos <= len(text) and (start_pos <= position < end_pos):
                    potential_abbrev = text[start_pos:end_pos]
                    if potential_abbrev.lower() == abbrev.lower():
                        if start_pos == 0 or not text[start_pos - 1].isalnum():
                            return True
        return False
    while i < len(paragraph):
        char = paragraph[i]
        current += char
        if char == '$':
            if not in_math_block:
                if i + 1 < len(paragraph) and paragraph[i + 1] == '$':
                    math_delimiter = '$$'
                    current += '$'
                    i += 1
                else:
                    math_delimiter = '$'
                in_math_block = True
            elif math_delimiter == '$$' and i + 1 < len(paragraph) and (paragraph[i + 1] == '$'):
                current += '$'
                i += 1
                in_math_block = False
                math_delimiter = None
            elif math_delimiter == '$':
                in_math_block = False
                math_delimiter = None
        elif not in_math_block:
            if char == '.' and i + 2 < len(paragraph) and (paragraph[i + 1:i + 3] == '..'):
                current += paragraph[i + 1:i + 3]
                i += 2
                context_before = current[-20:] if len(current) >= 20 else current
                context_after = paragraph[i + 1:i + 21] if i + 1 < len(paragraph) else ''
                math_indicators = ['+', '-', '*', '/', '=', '(', ')', '[', ']', 'g(', 'f(', 'h(', 'times', 'integer', 'induction']
                is_math_context = any((indicator in context_before.lower() or indicator in context_after.lower() for indicator in math_indicators))
                if not is_math_context:
                    splits.append(current.strip())
                    current = ''
            elif char in '.?!':
                if char == '.' and is_abbreviation_context(paragraph, i):
                    pass
                elif char == '.' and i > 0 and (i < len(paragraph) - 1):
                    prev_char = paragraph[i - 1]
                    next_char = paragraph[i + 1]
                    if prev_char.isdigit() and next_char.isdigit():
                        pass
                    elif i == 1 and prev_char.isdigit():
                        pass
                    else:
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
    if not sentence or not sentence.strip():
        return False
    cleaned = sentence.strip()
    if all((c == '-' for c in cleaned)):
        return False
    alphanumeric_chars = ''.join((c for c in cleaned if c.isalnum()))
    return bool(alphanumeric_chars)

def process_section(section_text, section_type):
    paragraphs = split_response_into_paragraphs(section_text)
    sentences = []
    for paragraph in paragraphs:
        paragraph_sentences = split_paragraph_into_sentences(paragraph)
        for sentence in paragraph_sentences:
            if is_valid_sentence(sentence):
                sentences.append({'sentence': sentence, 'type': section_type})
    return sentences

def process_response_to_sentences(response, apply_merging=True):
    if '</think>' in response:
        parts = response.split('</think>', 1)
        thinking_part = parts[0].strip()
        answer_part = parts[1].strip()
        thinking_sentences = process_section(thinking_part, 'think')
        answer_sentences = process_section(answer_part, 'answer')
        all_sentences = thinking_sentences + answer_sentences
    else:
        all_sentences = process_section(response, 'answer')
    if apply_merging:
        processed_sentences = merge_colon_and_equals_sentences(all_sentences)
    else:
        processed_sentences = all_sentences
    result = []
    for i, sentence_data in enumerate(processed_sentences):
        result.append({'id': str(i), 'sentence': sentence_data['sentence'], 'type': sentence_data['type']})
    return [row['sentence'] for row in result]

def merge_colon_and_equals_sentences(sentences):
    merged = []
    i = 0
    while i < len(sentences):
        current_sentence = sentences[i].copy()
        current_sentence['sentence'] = current_sentence['sentence'].replace('<think>', '').strip()
        while current_sentence['sentence'].rstrip().endswith(':') and i + 1 < len(sentences):
            next_sentence = sentences[i + 1].copy()
            next_sentence['sentence'] = next_sentence['sentence'].replace('<think>', '').strip()
            if current_sentence['type'] == next_sentence['type']:
                current_sentence['sentence'] = current_sentence['sentence'] + ' ' + next_sentence['sentence']
                i += 1
            else:
                break
        merged.append(current_sentence)
        i += 1
    final_merged = []
    for i, sentence in enumerate(merged):
        sentence_copy = sentence.copy()
        sentence_copy['sentence'] = sentence_copy['sentence'].replace('<think>', '').strip()
        if sentence_copy['sentence'].lstrip().startswith('=') and len(final_merged) > 0 and (final_merged[-1]['type'] == sentence_copy['type']):
            final_merged[-1]['sentence'] = final_merged[-1]['sentence'] + ' ' + sentence_copy['sentence']
        else:
            final_merged.append(sentence_copy)
    return final_merged
