import collections
from functools import partial


def get_text(paper, text_key):
    body_dict = collections.defaultdict(list)
    for rec in paper[text_key]:
        body_dict[rec['section']].append(rec['text'])

    body = ''
    for section, text_sections in body_dict.items():
        body += section + '\n\n'
        for text in text_sections:
            body += text + '\n\n'
    return body


def author_name(author_json):
    first = author_json.get('first')
    middle = "".join(author_json.get('middle'))
    last = author_json.get('last')
    if middle:
        return ' '.join([first, middle, last])
    return ' '.join([first, last])


def get_affiliation(author_json):
    affiliation = author_json['affiliation']
    institution = affiliation.get('institution', '')
    location = affiliation.get('location')
    if location:
        location = ' '.join(location.values())
    return f'{institution}, {location}'


def get_authors(paper, include_affiliation=False):
    if include_affiliation:
        return [f'{author_name(a)}, {get_affiliation(a)}'
                for a in paper['metadata']['authors']]
    else:
        return [author_name(a) for a in paper['metadata']['authors']]
