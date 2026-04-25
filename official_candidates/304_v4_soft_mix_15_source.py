from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.0698829989184774, '009': 0.401827243781245, '010': 0.15432495594497092, '011': 0.13985675497840735, '023': 0.23410804637689928}, '036': {'008': 0.7029972545964711, '009': 0.0698829989184774, '010': 0.0698829989184774, '011': 0.0698829989184774, '023': 0.08735374864809674}, '016': {'008': 0.0698829989184774, '009': 0.1956723969717367, '010': 0.0698829989184774, '011': 0.3127264201601863, '023': 0.3518351850311221}, '024': {'008': 0.08735374864809675, '009': 0.15416613094742893, '010': 0.43177710046059253, '011': 0.0698829989184774, '023': 0.2568200210254044}, '019': {'008': 0.0698829989184774, '009': 0.1784512293811119, '010': 0.2741319457574818, '011': 0.4076508270244515, '023': 0.0698829989184774}}, 'target_std': 165}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
