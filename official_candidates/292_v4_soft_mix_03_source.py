from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.09523809523809523, '009': 0.5238095238095238, '010': 0.09523809523809523, '011': 0.09523809523809523, '023': 0.19047619047619047}, '036': {'008': 0.6190476190476191, '009': 0.09523809523809523, '010': 0.09523809523809523, '011': 0.09523809523809523, '023': 0.09523809523809523}, '016': {'008': 0.09523809523809523, '009': 0.19047619047619047, '010': 0.09523809523809523, '011': 0.23809523809523808, '023': 0.38095238095238093}, '024': {'008': 0.09523809523809523, '009': 0.09523809523809523, '010': 0.47619047619047616, '011': 0.09523809523809523, '023': 0.23809523809523808}, '019': {'008': 0.09523809523809523, '009': 0.09523809523809523, '010': 0.23809523809523808, '011': 0.47619047619047616, '023': 0.09523809523809523}}, 'target_std': 177}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
