"""Summarize a complete (T+1) x T continual-learning score matrix."""
import argparse,json,math

def summarize(matrix):
    tasks=len(matrix)-1
    if tasks<2 or any(len(row)!=tasks for row in matrix):
        raise ValueError('Expected a complete (T+1) x T matrix with T >= 2')
    if any(not isinstance(v,(int,float)) or not math.isfinite(v) or not 0<=v<=100 for row in matrix for v in row):
        raise ValueError('All task scores must be finite numbers on the 0-100 scale')
    final=matrix[tasks]
    return {
        'AP':sum(final)/tasks,
        'ForgetRate':sum(max(matrix[k][j] for k in range(j+1,tasks))-final[j] for j in range(tasks-1))/(tasks-1),
        'BWT':sum(final[j]-matrix[j+1][j] for j in range(tasks-1))/(tasks-1),
        'FWT':sum(matrix[j][j]-matrix[0][j] for j in range(1,tasks))/(tasks-1),
        'tasks':tasks,
        'forget_rate_unit':'score points, not relative percent; negative values retained',
    }

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('matrix_json');args=p.parse_args()
    with open(args.matrix_json,encoding='utf-8') as f:matrix=json.load(f)
    print(json.dumps(summarize(matrix),ensure_ascii=False,indent=2))
