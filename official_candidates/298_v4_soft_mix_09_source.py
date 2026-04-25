from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.13843318800844984, '009': 0.30832846420063825, '010': 0.17146838060137534, '011': 0.15461369050294396, '023': 0.22715627668659266}, '036': {'008': 0.4462672479662007, '009': 0.13843318800844984, '010': 0.13843318800844984, '011': 0.13843318800844984, '023': 0.13843318800844984}, '016': {'008': 0.13843318800844984, '009': 0.1950649467391793, '010': 0.13843318800844984, '011': 0.26019146927951814, '023': 0.2678772079644029}, '024': {'008': 0.13843318800844984, '009': 0.1610858915007416, '010': 0.3339475931502539, '011': 0.13843318800844984, '023': 0.2281001393321048}, '019': {'008': 0.13843318800844984, '009': 0.19708750955099108, '010': 0.21771765023147108, '011': 0.3083284642006382, '023': 0.13843318800844984}}, 'target_std': 181}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
