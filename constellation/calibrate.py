"""Threshold-only scene-held-out analysis of frozen localization diagnostics.

This is not fully held-out model evaluation: feature development may already
have used the same scenes. Identification is omitted from this ablation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .data import DEFAULT_DATA, read_rows
from .metrics import score_scene


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('diagnostics',type=Path)
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path,default=Path('outputs/calibration.json'))
    parser.add_argument('--relative-gap', action='store_true',
                        help='Gate by (best score - next location)/(1 - best score), rather than raw difference')
    args=parser.parse_args()
    _,truth=read_rows(args.data/'train_ground_truth.csv')
    patches=json.loads(args.diagnostics.read_text(encoding='utf-8'))
    scene_ids={p['scene'] for p in patches}
    truth=[r for r in truth if r['Id'] in scene_ids]
    if len(truth)<2:
        raise ValueError('At least two fully evaluated labelled scenes are required')
    all_results=[]
    gates = (0,.04,.08,.12,.16,.20,.30,.50) if args.relative_gap else (0,.003,.005,.01,.015,.02,.025,.03,.05)
    for threshold in (.60,.65,.70,.72,.75,.80,.85,.90):
        for margin in gates:
            scores={}
            for row in truth:
                diagnostics=[p for p in patches if p['scene']==row['Id']]
                if len(diagnostics)!=int(row['n_patches']):
                    raise ValueError('Partial patch diagnostics cannot be calibrated')
                pred={k:'-1' for k in row}
                pred.update(Id=row['Id'],n_patches=row['n_patches'],constellation='unknown')
                for patch in diagnostics:
                    relative = patch['margin'] / max(1.0 - patch['score'], 1e-8)
                    margin_value = relative if args.relative_gap else patch['margin']
                    if patch['score']>=threshold and margin_value>=margin:
                        pred[patch['patch']]=str((patch['x'],patch['y'],0))
                scores[row['Id']]=score_scene(row,pred)['score']
            all_results.append(dict(threshold=threshold,margin=margin,scores=scores,mean=float(np.mean(list(scores.values())))))
    folds=[]
    for row in truth:
        test=row['Id']
        selected=max(all_results,key=lambda r:np.mean([v for k,v in r['scores'].items() if k!=test]))
        folds.append(dict(held_out_scene=test,threshold=selected['threshold'],margin=selected['margin'],
                          held_out_localization_only_score=selected['scores'][test]))
    result=dict(note=__doc__,gate='relative_gap' if args.relative_gap else 'raw_margin',
                threshold_only_folds=folds,
                development_best=max(all_results,key=lambda r:r['mean']),grid=all_results)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='grid'},indent=2))


if __name__=='__main__':
    main()
