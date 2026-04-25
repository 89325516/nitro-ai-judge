from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.15384615384615385, '009': 0.34265734265734266, '010': 0.15384615384615385, '011': 0.15384615384615385, '023': 0.1958041958041958}, '036': {'008': 0.3846153846153846, '009': 0.15384615384615385, '010': 0.15384615384615385, '011': 0.15384615384615385, '023': 0.15384615384615385}, '016': {'008': 0.15384615384615385, '009': 0.1958041958041958, '010': 0.15384615384615385, '011': 0.21678321678321677, '023': 0.2797202797202797}, '024': {'008': 0.15384615384615385, '009': 0.15384615384615385, '010': 0.3216783216783216, '011': 0.15384615384615385, '023': 0.21678321678321677}, '019': {'008': 0.15384615384615385, '009': 0.15384615384615385, '010': 0.21678321678321677, '011': 0.3216783216783216, '023': 0.15384615384615385}}, 'target_std': 169}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
