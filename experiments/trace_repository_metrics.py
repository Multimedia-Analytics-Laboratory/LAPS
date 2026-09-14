"""Exact metric definitions used by OnPolicyReplay/tools/eval.py."""
import re
from difflib import SequenceMatcher
from rouge_score import rouge_scorer
from trace_paper_metrics import _sari_sentence, _simplification_source


def _code(text):
    text=text.replace('<NUM_LIT>','0').replace('<STR_LIT>','').replace('<CHAR_LIT>','')
    for kind,value in re.findall(r'<(STR|NUM|CHAR)_LIT:(.*?)>',text,re.S):
        text=text.replace(f'<{kind}_LIT:{value}>',value)
    return text


def _fuzz_ratio(left, right):
    """Dependency-free equivalent of fuzzywuzzy.fuzz.ratio."""
    return int(round(100 * SequenceMatcher(None, left, right).ratio()))


def repository_scores_per_example(task, predictions, rows):
    """Return additive per-example scores whose mean is repository_score."""
    gold = [str(x['answer']) for x in rows]
    if task in {'C-STANCE', 'FOMC', 'ScienceQA'}:
        return [
            100.0 * float(bool(p) and g[:1] == p[:1])
            for g, p in zip(gold, predictions)
        ]
    if task == 'MeetingBank':
        scorer = rouge_scorer.RougeScorer(['rougeL'])
        return [
            100.0 * scorer.score(g, p)['rougeL'].fmeasure
            for g, p in zip(gold, predictions)
        ]
    if task == 'Py150':
        return [
            float(_fuzz_ratio(_code(p), _code(g)))
            for g, p in zip(gold, predictions)
        ]
    if task in {'NumGLUE-cm', 'NumGLUE-ds'}:
        result = []
        for g, p in zip(gold, predictions):
            values = re.findall(r'(\-?[0-9\.\,]+)', p)
            valid = [x for x in values if x not in {'', '.'}]
            result.append(100.0 * float(bool(valid) and valid[-1] == g))
        return result
    if task == '20Minuten':
        return [
            float(_sari_sentence(
                _simplification_source(str(row['prompt'])), prediction, [answer],
            ))
            for row, prediction, answer in zip(rows, predictions, gold)
        ]
    raise ValueError(task)


def repository_score(task, predictions, rows):
    gold=[str(x['answer']) for x in rows]
    if task in {'C-STANCE','FOMC','ScienceQA'}:
        return 100*sum(bool(p) and g[:1]==p[:1] for g,p in zip(gold,predictions))/len(rows)
    if task=='MeetingBank':
        s=rouge_scorer.RougeScorer(['rougeL'])
        return sum(100*s.score(g,p)['rougeL'].fmeasure for g,p in zip(gold,predictions))/len(rows)
    if task=='Py150':
        return sum(_fuzz_ratio(_code(p),_code(g)) for g,p in zip(gold,predictions))/len(rows)
    if task in {'NumGLUE-cm','NumGLUE-ds'}:
        good=0
        for g,p in zip(gold,predictions):
            values=re.findall(r'(\-?[0-9\.\,]+)',p)
            valid=[x for x in values if x not in {'','.'}]
            good+=bool(valid) and valid[-1]==g
        return 100*good/len(rows)
    if task=='20Minuten':
        return sum(_sari_sentence(_simplification_source(str(r['prompt'])),p,[g]) for r,p,g in zip(rows,predictions,gold))/len(rows)
    raise ValueError(task)
