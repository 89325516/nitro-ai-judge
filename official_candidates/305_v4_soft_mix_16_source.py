from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.030368464124786776, '009': 0.4631190779029984, '010': 0.14045414657713887, '011': 0.12159217519054241, '023': 0.24446613620453359}, '036': {'008': 0.8557497954072628, '009': 0.030368464124786772, '010': 0.030368464124786772, '011': 0.030368464124786772, '023': 0.05314481221837686}, '016': {'008': 0.030368464124786772, '009': 0.19435817039863534, '010': 0.030368464124786772, '011': 0.346959702625689, '023': 0.39794519872610207}, '024': {'008': 0.05314481221837686, '009': 0.1402470888671971, '010': 0.5021642460634386, '011': 0.030368464124786772, '023': 0.2740753887262007}, '019': {'008': 0.030368464124786776, '009': 0.17190719870638232, '010': 0.29664467910984904, '011': 0.47071119393419514, '023': 0.030368464124786776}}, 'target_std': 169}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
