import numpy as np
from sympy.utilities.iterables import multiset_permutations

seq = [1.0, 1.0, 1.1, 4.3, 0.9, 1.3, 1.0, 1.0]
np.random.shuffle(seq)

# seq = np.random.rand(1,4)
# seq = seq.tolist()[0]

n = 5
    
def sequential_median(seq):
    
    # if len(seq) <= n:
    #     return seq
    
    buffer = []
    for data in seq:                        
        if len(buffer) >= n and (data < buffer[0] or data > buffer[-1]):
            continue
        else:
            insertIdx = len(buffer)
            for idx in range(len(buffer)):
                if data < buffer[idx]:
                    insertIdx = idx
                    break
            buffer = np.hstack((np.hstack((buffer[:insertIdx], data)), buffer[insertIdx:]))
            
            if len(buffer) <= n:
                continue
            
            if insertIdx > len(buffer)/2:
                buffer = buffer[1:]
            else:
                buffer = buffer[:-1]
    return buffer

def sequential_median_impl2(seq):
    
    v = []
    for value in seq:    
        
        if len(v) < n:
            v.append(value)
            v = np.sort(v).tolist()
            continue
        
        if value < v[0]:
            continue

        if value >= v[n-1]:
            continue
        

        insertIndex = 0
        buffer = v.copy()
        while insertIndex < n:
            if (value < v[insertIndex]):
                break
            insertIndex += 1

        buffer[insertIndex] = value
        while insertIndex < n-1:
            buffer[insertIndex+1] = v[insertIndex]
            insertIndex += 1
        v = buffer        
    return v

# buffer = sequential_median(seq) 
# print(buffer)
# exit(0)

error = []
for seq_case in multiset_permutations(seq):
    for k in range(1, len(seq_case)):
        buffer = sequential_median(seq_case[:k]) 
        #buffer = sequential_median_impl2(seq_case[:k]) 
        print(np.median(buffer), np.median(seq_case[:k]))
        error.append(np.median(buffer) - np.median(seq_case[:k]))

print(np.mean(np.abs(error)))
print(np.median(np.abs(error)))
print(np.percentile(np.abs(error), 99))
