from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.12588742492762836, '009': 0.3106863289906573, '010': 0.1802983934825779, '011': 0.16568593681505114, '023': 0.21744191578408534}, '036': {'008': 0.4369311742817221, '009': 0.12588742492762836, '010': 0.14468250550248754, '011': 0.13232485006597297, '023': 0.1601740452221889}, '016': {'008': 0.12588742492762836, '009': 0.20569643833868462, '010': 0.12588742492762836, '011': 0.26376012130797305, '023': 0.2787685904980856}, '024': {'008': 0.17804949363442743, '009': 0.16729502420134412, '010': 0.31104003366858823, '011': 0.12588742492762836, '023': 0.21772802356801177}, '019': {'008': 0.13324448222859364, '009': 0.19043478354168564, '010': 0.23809164241871789, '011': 0.3123416668833744, '023': 0.12588742492762836}}, 'target_std': 173}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
