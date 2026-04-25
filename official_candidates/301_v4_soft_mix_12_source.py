from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.03394350830403685, '009': 0.49218087040853437, '010': 0.12304521760213359, '011': 0.07758516183779852, '023': 0.27324524184749666}, '036': {'008': 0.8642259667838527, '009': 0.03394350830403685, '010': 0.03394350830403685, '011': 0.03394350830403685, '023': 0.03394350830403685}, '016': {'008': 0.03394350830403685, '009': 0.1866892956722027, '010': 0.03394350830403685, '011': 0.36234695114559345, '023': 0.3830767365741302}, '024': {'008': 0.03394350830403684, '009': 0.09504182325130317, '010': 0.5612801551703237, '011': 0.03394350830403684, '023': 0.27579100497029946}, '019': {'008': 0.03394350830403685, '009': 0.19214450236392291, '010': 0.24778761061946902, '011': 0.49218087040853437, '023': 0.03394350830403685}}, 'target_std': 173}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
