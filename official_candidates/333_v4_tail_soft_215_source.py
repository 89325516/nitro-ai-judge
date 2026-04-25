from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 1.0, 'target_std': 215, 'tail_boost': 0.22, 'tail_quantile': 0.88, 'weights': {'040': {'008': 0.07602497963616617, '009': 0.4181373879989139, '010': 0.14254683681781158, '011': 0.10860711376595167, '023': 0.2546836817811567}, '036': {'008': 0.6959000814553353, '009': 0.07602497963616617, '010': 0.07602497963616617, '011': 0.07602497963616617, '023': 0.07602497963616617}, '016': {'008': 0.07602497963616617, '009': 0.19006244909041542, '010': 0.07602497963616617, '011': 0.321205538962802, '023': 0.33668205267445017}, '024': {'008': 0.07602497963616617, '009': 0.12163996741786588, '010': 0.469725767037741, '011': 0.07602497963616617, '023': 0.2565843062720608}, '019': {'008': 0.07602497963616617, '009': 0.1941352158566386, '010': 0.23567743687211512, '011': 0.4181373879989139, '023': 0.07602497963616617}}}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
