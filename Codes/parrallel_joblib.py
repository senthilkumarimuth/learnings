from joblib import Parallel, delayed
import os
print(os.getcwd())
def dir(k):
    print('printed from process',os.getcwd())
    return os.getcwd()

ls5=[1,2,3,4]

df_top10 = Parallel(n_jobs=4, verbose=0, prefer="processes")(delayed(dir)(k) for k in ls5)


#usecase 2

s = [1, 2, 3, 4]
k = [0, 1, 2, 3]
s1 = [1, 2, 3, 4]


def sqaure(a, index):
    return a+s[index]


result = Parallel(n_jobs=4, verbose=0, prefer="processes")
(delayed(sqaure)(a, index) for a, index in zip(s1, k))

print(result)
