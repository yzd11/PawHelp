"""Conservative lexical near-duplicate isolation before splitting, not relabelling."""
import argparse
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from app.topics.data import digest, read_jsonl, text_key, write_json, write_jsonl


def near_key(text):
    # Group-only canonical spelling; preserve original training text and labels.
    return text_key(text).replace('优惠卷','优惠券').replace('优会券','优惠券')


def group_near_duplicates(rows, threshold=.85):
    if not .8 <= threshold <= 1:
        raise ValueError('near-duplicate threshold must be .8..1')
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    def union(a,b):
        parent[root(b)]=root(a)
    keys=[near_key(row['text']) for row in rows]
    index=defaultdict(list)
    pairs=set()
    for i,key in enumerate(keys):
        grams={key[j:j+3] for j in range(max(0,len(key)-2))}
        candidates={previous for gram in grams for previous in index[gram]}
        for previous in candidates:
            a,b=keys[previous],key
            if min(len(a),len(b))<6 or 2*min(len(a),len(b))/(len(a)+len(b))<threshold:
                continue
            matcher=SequenceMatcher(None,a,b,autojunk=False)
            if matcher.quick_ratio()<threshold:
                continue
            score=matcher.ratio()
            if score>=threshold:
                union(previous,i)
                pairs.add((previous,i,score))
        for gram in grams:
            index[gram].append(i)
    # Known lineage and exact duplicates must also participate in the component.
    known={}
    for i,row in enumerate(rows):
        for value in (('source',row['group_id']),('exact',text_key(row['text']))):
            if value in known: union(known[value],i)
            else: known[value]=i
    components=defaultdict(list)
    for i in range(len(rows)): components[root(i)].append(i)
    output=[dict(row) for row in rows]
    for members in components.values():
        group='near-'+digest(sorted(rows[i]['id'] for i in members))[:20]
        for i in members: output[i]['near_duplicate_group_id']=group
    evidence=[{'a':rows[a]['id'],'b':rows[b]['id'],'similarity':round(score,4)}
              for a,b,score in sorted(pairs)]
    return output, {'policy':'shared character trigram, min length 6, SequenceMatcher >= threshold; group only, no label merge',
        'threshold':threshold,'input_sha256':digest(rows),'output_sha256':digest(output),
        'groups':len(components),'linked_pairs':evidence,
        'limitations':'lexical grouping is conservative; unrecognised semantic paraphrases may remain'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.groups.json').exists():
        raise FileExistsError('grouped corpora are versioned; never overwrite')
    output,report=group_near_duplicates(read_jsonl(args.input))
    write_jsonl(args.output,output)
    write_json(args.output.with_suffix('.groups.json'),report)
    print({'samples':len(output),'groups':report['groups'],'near_pairs':len(report['linked_pairs'])})


if __name__=='__main__': main()
