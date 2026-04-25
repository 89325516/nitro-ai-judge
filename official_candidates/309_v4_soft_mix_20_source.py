from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.02675251948654516, '009': 0.45874323765149255, '010': 0.15394501286889892, '011': 0.11978655461415683, '023': 0.2407726753789065}, '036': {'008': 0.7538564671288123, '009': 0.02675251948654516, '010': 0.07068838802790973, '011': 0.04180081169772682, '023': 0.106901813659006}, '016': {'008': 0.02675251948654516, '009': 0.21331614222166279, '010': 0.02675251948654516, '011': 0.34904731569577474, '023': 0.38413150310947214}, '024': {'008': 0.1486879261702018, '009': 0.12354799908331768, '010': 0.45957006689386526, '011': 0.02675251948654516, '023': 0.24144148836607016}, '019': {'008': 0.04395056772789563, '009': 0.17764010155698176, '010': 0.28904401272278096, '011': 0.46261279850579645, '023': 0.02675251948654516}}, 'target_std': 165}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
