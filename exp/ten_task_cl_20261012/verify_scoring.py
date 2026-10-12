"""Validate fast scoring against an independent ordinary dynamic-programming LCS."""
import random
from run import score


def reference(a,b):
    if not a or not b:
        return 100.0 if a==b else 0.0
    previous=[0]*(len(b)+1)
    for left in a:
        current=[0]
        for index,right in enumerate(b):
            current.append(previous[index]+1 if left==right else max(previous[index+1],current[-1]))
        previous=current
    return 200*previous[-1]/(len(a)+len(b))


def main():
    rng=random.Random(42)
    vocabulary=['a','b','c','heart','肝','t0','t1','mri']
    for _ in range(500):
        a=[rng.choice(vocabulary) for _ in range(rng.randrange(81))]
        b=[rng.choice(vocabulary) for _ in range(rng.randrange(81))]
        result=score(' '.join(a),[' '.join(b)])['rougeL']
        assert abs(result-reference(a,b))<1e-10
    print('PASS: 500 independent DP comparisons including empty and Chinese sequences; scoring unchanged')


if __name__=='__main__':
    main()
