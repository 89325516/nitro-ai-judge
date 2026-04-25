from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.14559894109861019, '009': 0.3242885506287227, '010': 0.14559894109861019, '011': 0.14559894109861019, '023': 0.23891462607544672}, '036': {'008': 0.4176042356055592, '009': 0.14559894109861019, '010': 0.14559894109861019, '011': 0.14559894109861019, '023': 0.14559894109861019}, '016': {'008': 0.14559894109861019, '009': 0.18530774321641297, '010': 0.14559894109861019, '011': 0.25876902713434813, '023': 0.26472534745201853}, '024': {'008': 0.1455989410986102, '009': 0.16942422236929186, '010': 0.33421575115817337, '011': 0.1455989410986102, '023': 0.20516214427531437}, '019': {'008': 0.1455989410986102, '009': 0.1753805426869623, '010': 0.22898742554599605, '011': 0.3044341495698213, '023': 0.1455989410986102}}, 'target_std': 165}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
