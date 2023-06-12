import os

for ptb_rate in [0.05, 0.1, 0.15, 0.2, 0.25]:
    os.system(f'python refineAdj.py --ptb_rate {ptb_rate}')
