from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.06331831413272608, '009': 0.4041326189803733, '010': 0.16366542667738462, '011': 0.13671648838952044, '023': 0.23216715181999562}, '036': {'008': 0.6369589412284656, '009': 0.06331831413272607, '010': 0.09798103225410304, '011': 0.07519049803261221, '023': 0.1265512143520931}, '016': {'008': 0.06331831413272608, '009': 0.2105056233009051, '010': 0.06331831413272608, '011': 0.3175892331762259, '023': 0.34526851525741686}, '024': {'008': 0.15951790620205777, '009': 0.13968403845037752, '010': 0.40478493677707034, '011': 0.06331831413272608, '023': 0.23269480443776833}, '019': {'008': 0.07688652430402453, '009': 0.182359405135618, '010': 0.270250290158716, '011': 0.40718546626891544, '023': 0.06331831413272608}}, 'target_std': 181}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
