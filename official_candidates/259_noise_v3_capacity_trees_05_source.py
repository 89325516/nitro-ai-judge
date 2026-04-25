from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'capacity', 'target': 'median_of_means', 'model': 'trees', 'capacity': 768, 'weight': 0.34, 'target_mae': 8.75}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--neutral-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.noise_aware_v3_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.neutral_output,a.output,CONFIG)
if __name__=='__main__': main()
