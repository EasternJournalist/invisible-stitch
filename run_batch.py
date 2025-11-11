
from pathlib import Path
from tqdm import tqdm
import os
import time

if __name__ == "__main__":
    data_root = Path('/mnt/blob/data_v4/user_study_samples')
    output_root = Path('/mnt/blob/workspace/invisible_stitch/user_study_samples')

    metainfo_list = data_root.joinpath('.metainfo_list.txt').read_text().strip().splitlines()
    for metainfo_relpath in tqdm(metainfo_list, desc='Processing scenes'):
        metainfo_abspath = data_root / metainfo_relpath
        for seed in [0, 1]:
            output_path = output_root / Path(metainfo_relpath).parent / f'seed_{seed}'
            cmd = f"python run.py --metainfo_path {metainfo_abspath} --output_path {output_path} --seed {seed}"
            print(f"Running command: {cmd}")
            os.system(cmd)
            time.sleep(1)  