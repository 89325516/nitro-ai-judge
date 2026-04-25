from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.11017115876450916, '009': 0.35805626598465473, '010': 0.1583710407239819, '011': 0.13377926421404684, '023': 0.2396222703128074}, '036': {'008': 0.5593153649419634, '009': 0.11017115876450916, '010': 0.11017115876450916, '011': 0.11017115876450916, '023': 0.11017115876450916}, '016': {'008': 0.11017115876450917, '009': 0.192799527837891, '010': 0.11017115876450917, '011': 0.28782215227228014, '023': 0.29903600236081057}, '024': {'008': 0.11017115876450917, '009': 0.14322250639386191, '010': 0.395435766279756, '011': 0.11017115876450917, '023': 0.24099940979736376}, '019': {'008': 0.11017115876450916, '009': 0.19575054101908324, '010': 0.22585087546724372, '011': 0.35805626598465473, '023': 0.11017115876450916}}, 'target_std': 165}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
