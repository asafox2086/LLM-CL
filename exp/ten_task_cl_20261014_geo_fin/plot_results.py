"""Final actual result matrices -> reproducible PNG/PDF scientific figures."""
import argparse
import json
import sys
import os
import importlib.util
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if importlib.util.find_spec('matplotlib') is None:
    runtime=ROOT/'.plot-env/bin/python'
    if not runtime.exists():
        raise RuntimeError('Configured plotting runtime is unavailable')
    os.execv(str(runtime),[str(runtime),str(Path(__file__).resolve())]+sys.argv[1:])
import matplotlib
matplotlib.use('Agg')
import numpy as np

sys.path.insert(0,str(HERE/'plot_templates'))
from line_chart_template import plot_line_chart
from heatmap_template import plot_heatmap


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    config=json.loads((args.run_dir/'config.json').read_text())
    matrices=json.loads((args.run_dir/'score_matrices.json').read_text())
    output=args.output_dir or ROOT/'summary/ten_task_cl_20261014_geo_fin/figures'/config['method']/args.run_dir.name
    output.mkdir(parents=True,exist_ok=True)
    labels={'exact_match':'Exact-match accuracy','token_f1':'Token F1','rougeL':'ROUGE-L'}
    colors={'exact_match':'#7F7F7F','token_f1':'#4F81BD','rougeL':'#9BBB59'}
    series=[]
    for key,title in labels.items():
        array=np.asarray(matrices[key])
        if array.shape!=(11,10) or not np.isfinite(array).all():
            raise ValueError('Final plot requires complete real 11 x 10 matrices')
        plot_heatmap(array,[f'T{i}' for i in range(1,11)],[f'S{i}' for i in range(11)],
            title=config['method']+' — '+title,xlabel='Test task',ylabel='Completed training stage',
            out_base=str(output/(key+'_matrix')))
        series.append({'label':title,'values':array.mean(1),'color':colors[key]})
    plot_line_chart(np.arange(11),series,title=config['method']+' — all-task performance',
        xlabel='Completed training stage',ylabel='Mean score (0–100)',out_base=str(output/'mean_scores'))
    (output/'plotted_data.json').write_text(json.dumps({'config':config,'score_matrices':matrices},indent=2)+'\n')
    print('FIGURES_SAVED',output)


if __name__=='__main__':
    main()
