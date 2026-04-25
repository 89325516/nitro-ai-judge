from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.11869436201780416, '009': 0.3857566765578635, '010': 0.11869436201780416, '011': 0.11869436201780416, '023': 0.258160237388724}, '036': {'008': 0.5252225519287834, '009': 0.11869436201780416, '010': 0.11869436201780416, '011': 0.11869436201780416, '023': 0.11869436201780416}, '016': {'008': 0.11869436201780416, '009': 0.17804154302670622, '010': 0.11869436201780416, '011': 0.2878338278931751, '023': 0.29673590504451036}, '024': {'008': 0.11869436201780416, '009': 0.1543026706231454, '010': 0.400593471810089, '011': 0.11869436201780416, '023': 0.20771513353115725}, '019': {'008': 0.11869436201780416, '009': 0.1632047477744807, '010': 0.2433234421364985, '011': 0.35608308605341243, '023': 0.11869436201780416}}, 'target_std': 169}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
