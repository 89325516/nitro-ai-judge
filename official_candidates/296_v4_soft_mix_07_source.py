from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.08438818565400844, '009': 0.4641350210970464, '010': 0.08438818565400844, '011': 0.08438818565400844, '023': 0.28270042194092826}, '036': {'008': 0.6624472573839663, '009': 0.08438818565400844, '010': 0.08438818565400844, '011': 0.08438818565400844, '023': 0.08438818565400844}, '016': {'008': 0.08438818565400844, '009': 0.16877637130801687, '010': 0.08438818565400844, '011': 0.3248945147679325, '023': 0.33755274261603374}, '024': {'008': 0.08438818565400844, '009': 0.13502109704641352, '010': 0.48523206751054854, '011': 0.08438818565400844, '023': 0.2109704641350211}, '019': {'008': 0.08438818565400844, '009': 0.14767932489451477, '010': 0.2616033755274262, '011': 0.4219409282700422, '023': 0.08438818565400844}}, 'target_std': 173}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
