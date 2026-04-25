from __future__ import annotations
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
CONFIG={'mode': 'single', 'source': 'soft', 'zero_strategy': 'none', 'movement_weight': 0.95, 'weights': {'040': {'008': 0.09618520358718438, '009': 0.3550462751918424, '010': 0.17240254750810768, '011': 0.15193383200943525, '023': 0.22443214170343023}, '036': {'008': 0.5318864794252143, '009': 0.09618520358718438, '010': 0.12251282021008676, '011': 0.10520256642348291, '023': 0.14421293035403157}, '016': {'008': 0.09618520358718438, '009': 0.20797940951088553, '010': 0.09618520358718438, '011': 0.2893133723444354, '023': 0.3103368109703102}, '024': {'008': 0.1692523522860342, '009': 0.1541877960533956, '010': 0.35554173468834227, '011': 0.09618520358718441, '023': 0.2248329133850435}, '019': {'008': 0.10649076111438271, '009': 0.18660131565669205, '010': 0.2533576940062788, '011': 0.357365025635462, '023': 0.09618520358718438}}, 'target_std': 177}
def main():
    p=argparse.ArgumentParser(); p.add_argument('--train',type=Path,required=True); p.add_argument('--test',type=Path,required=True); p.add_argument('--best-output',type=Path,required=True); p.add_argument('--previous-best-output',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    from experiments.high_risk_reconstruction_v4_emit import runtime_write
    runtime_write(a.train,a.test,a.best_output,a.previous_best_output,a.output,CONFIG)
if __name__=='__main__': main()
