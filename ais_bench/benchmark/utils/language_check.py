"""Utilities for checking whether generated predictions are pure English.

During evaluation this is used to detect mixed Chinese/English (or any other
non-English script) in model outputs, e.g. for datasets like GPQA whose answers
are expected to be English-only.
"""

import re
from typing import Dict, List, Optional, Sequence, Set

# Script -> compiled pattern. Ordered so that detection result is deterministic.
_SCRIPT_PATTERNS = [
    ('chinese', re.compile(r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]')),
    ('fullwidth_cjk', re.compile(r'[\u3000-\u303f\uff00-\uffef]')),
    ('japanese_kana', re.compile(r'[\u3040-\u30ff]')),
    ('korean_hangul', re.compile(r'[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]')),
    ('greek', re.compile(r'[\u0370-\u03ff\u1f00-\u1fff]')),
    ('cyrillic', re.compile(r'[\u0400-\u04ff]')),
    ('latin_extended', re.compile(r'[\u00c0-\u024f]')),
    ('other_script', re.compile(
        r'[\u0590-\u05ff\u0600-\u06ff\u0750-\u077f\u0900-\u097f'
        r'\u0e00-\u0e7f\u0f00-\u0fff\u10a0-\u10ff]')),
]

# Any character above the ASCII range (covers math symbols, emoji, etc.).
_NON_ASCII_RE = re.compile(r'[^\x00-\x7f]')


def detect_scripts(text: Optional[str]) -> Set[str]:
    """Return the set of script categories detected in ``text``.

    If non-ASCII content is found that does not match any known script
    (e.g. math symbols, emoji, box-drawing chars), it is reported as
    ``other_non_latin``.
    """
    if not isinstance(text, str) or not text:
        return set()
    scripts = set()
    for name, pattern in _SCRIPT_PATTERNS:
        if pattern.search(text):
            scripts.add(name)
    if _NON_ASCII_RE.search(text) and not scripts:
        scripts.add('other_non_latin')
    return scripts


def has_chinese(text: Optional[str]) -> bool:
    """Whether the text contains Chinese ideographs or full-width CJK chars."""
    return bool(detect_scripts(text) & {'chinese', 'fullwidth_cjk'})


# Scripts that are NOT a real switch away from English writing. Latin-extended
# characters (accented letters like é/ü or symbols like ×/–) and generic
# non-Latin punctuation/symbols (math symbols, emoji) are common in legitimate
# technical English output, so they should not trigger a mixed-language alert.
_NOISE_SCRIPTS = frozenset({'latin_extended', 'other_non_latin'})


def significant_scripts(text: Optional[str]) -> Set[str]:
    """Detected scripts that indicate a genuine foreign writing system.

    Returns ``detect_scripts(text)`` minus noise categories (``latin_extended``
    and ``other_non_latin``). This is what should drive a mixed-language
    warning: pure-English technical content with math symbols (``×``, ``–``)
    yields an empty set, while real foreign scripts (Chinese, Cyrillic, etc.)
    are still reported.
    """
    return detect_scripts(text) - _NOISE_SCRIPTS


# One or more consecutive Chinese ideographs or full-width CJK characters.
_CHINESE_RE = re.compile(
    r'[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\u3000-\u303f\uff00-\uffef]+')


def chinese_context(text: Optional[str], context: int = 20,
                    max_spans: int = 5) -> List[str]:
    """Return context snippets around each Chinese run in ``text``.

    The Chinese characters are wrapped in ``[[..]]`` markers, and each snippet
    shows up to ``context`` characters before and after the run so the mixed
    region is easy to spot. At most ``max_spans`` snippets are returned to keep
    the warning log concise.
    """
    if not isinstance(text, str) or not text:
        return []
    snippets = []
    for match in _CHINESE_RE.finditer(text):
        start, end = match.start(), match.end()
        lo, hi = max(0, start - context), min(len(text), end + context)
        snippets.append(
            text[lo:start] + '[[' + text[start:end] + ']]' + text[end:hi]
        )
        if len(snippets) >= max_spans:
            break
    return snippets


def is_pure_ascii(text: Optional[str]) -> bool:
    """Whether the text consists only of ASCII characters."""
    if not isinstance(text, str):
        return False
    return not bool(_NON_ASCII_RE.search(text))


def language_check(preds: Sequence,
                   pred_ids: Optional[Sequence] = None) -> Dict:
    """Scan predictions and produce a language-mix report.

    Args:
        preds: raw prediction strings (or lists of strings when self-consistency
            is used), aligned with the samples being evaluated.
        pred_ids: optional sample ids aligned with ``preds``.

    Returns:
        A dict report::

            {
                'total': int,          # number of predictions checked
                'pure_ascii': int,     # ASCII-only predictions
                'has_chinese': int,    # predictions containing Chinese chars
                'has_non_ascii': int,  # predictions containing any non-ASCII char
                'details': [           # only entries that contain non-ASCII chars
                    {
                        'id': id, 'prediction': str, 'scripts': [...],
                        'has_chinese': bool,
                    }, ...
                ],
            }
    """
    total = 0
    pure_ascii_cnt = 0
    has_chinese_cnt = 0
    has_non_ascii_cnt = 0
    details = []
    for i, pred in enumerate(preds):
        # self-consistency may produce a list of predictions per sample
        if isinstance(pred, list):
            pred = '\n'.join(str(p) for p in pred)
        if not isinstance(pred, str):
            continue
        total += 1
        scripts = detect_scripts(pred)
        non_ascii = not is_pure_ascii(pred)
        if non_ascii:
            has_non_ascii_cnt += 1
        if scripts & {'chinese', 'fullwidth_cjk'}:
            has_chinese_cnt += 1
        if non_ascii:
            details.append({
                'id': pred_ids[i] if pred_ids is not None else i,
                'prediction': pred,
                'scripts': sorted(scripts),
                'has_chinese': bool(scripts & {'chinese', 'fullwidth_cjk'}),
            })
        else:
            pure_ascii_cnt += 1
    return {
        'total': total,
        'pure_ascii': pure_ascii_cnt,
        'has_chinese': has_chinese_cnt,
        'has_non_ascii': has_non_ascii_cnt,
        'details': details,
    }
