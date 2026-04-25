from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.10357525179210968, '009': 0.34956647479837016, '010': 0.16615196641650923, '011': 0.1554301343451724, '023': 0.22527617264783853}, '036': {'008': 0.5727520863575477, '009': 0.10357525179210968, '010': 0.10357525179210968, '011': 0.10357525179210968, '023': 0.11652215826612337}, '016': {'008': 0.10357525179210968, '009': 0.1967929784050084, '010': 0.10357525179210968, '011': 0.2835372517809002, '023': 0.31251926622987203}, '024': {'008': 0.11652215826612339, '009': 0.16603426726674553, '010': 0.37176117161096506, '011': 0.10357525179210968, '023': 0.24210715106405634}, '019': {'008': 0.10357525179210968, '009': 0.1840310277377663, '010': 0.2549363583883063, '011': 0.35388211028970806, '023': 0.10357525179210968}}, 'target_std': 181}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
