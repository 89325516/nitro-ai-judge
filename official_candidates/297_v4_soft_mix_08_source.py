from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.03913894324853229, '009': 0.5675146771037183, '010': 0.03913894324853229, '011': 0.03913894324853229, '023': 0.31506849315068497}, '036': {'008': 0.8434442270058709, '009': 0.03913894324853228, '010': 0.03913894324853228, '011': 0.03913894324853228, '023': 0.03913894324853228}, '016': {'008': 0.03913894324853228, '009': 0.15655577299412912, '010': 0.03913894324853228, '011': 0.37377690802348335, '023': 0.3913894324853229}, '024': {'008': 0.03913894324853228, '009': 0.10958904109589039, '010': 0.5968688845401174, '011': 0.03913894324853228, '023': 0.2152641878669276}, '019': {'008': 0.03913894324853229, '009': 0.12720156555772993, '010': 0.28571428571428575, '011': 0.5088062622309198, '023': 0.03913894324853229}}, 'target_std': 177}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
