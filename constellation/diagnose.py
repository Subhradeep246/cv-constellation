"""Training-only oracle probes; NEVER called by the prediction pipeline."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

from .data import DEFAULT_DATA, discover_scenes, read_gray, read_rows, row_points
from .localize import Localizer, LocalizerConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output', type=Path, default=Path('outputs/oracle-localization.json'))
    args = parser.parse_args()
    cv2.setNumThreads(8)
    loc=Localizer(LocalizerConfig())
    _, rows=read_rows(args.data/'train_ground_truth.csv')
    scenes={s.scene_id:s for s in discover_scenes(args.data/'train')}
    results=[]
    for row in rows:
        scene=scenes[row['Id']]
        sky=read_gray(scene.image_path)
        filtered=loc.filtered(sky)
        sources=loc.detect(sky)
        tree=cKDTree(sources)
        for path,gt in zip(scene.patches,row_points(row)):
            if gt is None:
                continue
            distance,index=tree.query(gt[:2])
            proposals=np.array([[gt[0],gt[1],s] for s in loc.config.scales])
            match=loc.verify(filtered,read_gray(path),proposals)
            results.append(dict(scene=scene.scene_id,patch=path.stem,source_distance=float(distance),
                                oracle_score=match.score,oracle_scale=match.scale,
                                oracle_error=float(np.linalg.norm(np.array([match.x,match.y])-gt[:2]))))
        selected=[r for r in results if r['scene']==scene.scene_id]
        print(scene.scene_id, 'n=',len(selected), 'source <=3px=',sum(r['source_distance']<=3 for r in selected),
              'oracle score median=',round(float(np.median([r['oracle_score'] for r in selected])),3),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
