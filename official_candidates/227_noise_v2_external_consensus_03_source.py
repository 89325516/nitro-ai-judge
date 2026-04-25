from __future__ import annotations
import argparse,csv,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'external', 'prior': 'text', 'strength': 18.0, 'weight': 0.08, 'model': 'hgb', 'gamma': 0.35}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--anchor-output',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.noise_aware_v2_emit import runtime_write
    runtime_write(a.train,a.test,a.anchor_output,a.best_output,a.output,CONFIG)
if __name__=='__main__': main()
