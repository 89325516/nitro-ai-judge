from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.13264375286292043, '009': 0.3044777054353401, '010': 0.17635589869274648, '011': 0.1688663030839281, '023': 0.2176563399250649}, '036': {'008': 0.46038109630766455, '009': 0.13264375286292043, '010': 0.13264375286292043, '011': 0.13264375286292043, '023': 0.1416876451035741}, '016': {'008': 0.13264375286292043, '009': 0.19775977699562683, '010': 0.13264375286292043, '011': 0.25835385500800634, '023': 0.27859886227052594}, '024': {'008': 0.1416876451035741, '009': 0.17627368149055872, '010': 0.31998152070503205, '011': 0.13264375286292043, '023': 0.22941339983791464}, '019': {'008': 0.13264375286292043, '009': 0.18884508321555393, '010': 0.23837507487638054, '011': 0.3074923361822246, '023': 0.13264375286292043}}, 'target_std': 177}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
