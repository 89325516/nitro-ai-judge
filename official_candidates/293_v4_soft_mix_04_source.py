from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.04651162790697674, '009': 0.6744186046511628, '010': 0.04651162790697674, '011': 0.04651162790697674, '023': 0.18604651162790697}, '036': {'008': 0.813953488372093, '009': 0.04651162790697674, '010': 0.04651162790697674, '011': 0.04651162790697674, '023': 0.04651162790697674}, '016': {'008': 0.04651162790697674, '009': 0.18604651162790697, '010': 0.04651162790697674, '011': 0.2558139534883721, '023': 0.46511627906976744}, '024': {'008': 0.04651162790697674, '009': 0.04651162790697674, '010': 0.6046511627906977, '011': 0.04651162790697674, '023': 0.2558139534883721}, '019': {'008': 0.04651162790697674, '009': 0.04651162790697674, '010': 0.2558139534883721, '011': 0.6046511627906977, '023': 0.04651162790697674}}, 'target_std': 181}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
