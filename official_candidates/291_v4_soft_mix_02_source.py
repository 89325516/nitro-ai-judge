from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.12903225806451613, '009': 0.41935483870967744, '010': 0.12903225806451613, '011': 0.12903225806451613, '023': 0.1935483870967742}, '036': {'008': 0.48387096774193544, '009': 0.12903225806451613, '010': 0.12903225806451613, '011': 0.12903225806451613, '023': 0.12903225806451613}, '016': {'008': 0.12903225806451613, '009': 0.1935483870967742, '010': 0.12903225806451613, '011': 0.2258064516129032, '023': 0.3225806451612903}, '024': {'008': 0.12903225806451613, '009': 0.12903225806451613, '010': 0.3870967741935484, '011': 0.12903225806451613, '023': 0.2258064516129032}, '019': {'008': 0.12903225806451613, '009': 0.12903225806451613, '010': 0.2258064516129032, '011': 0.3870967741935484, '023': 0.12903225806451613}}, 'target_std': 173}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
