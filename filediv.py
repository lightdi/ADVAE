import os 


dir_list = os.listdir('/media/lightdi/CRUCIAL/Datasets/CAS-PEAL.VDGAN/')


files = [
        'LoadPEAL100.txt',
        'LoadPEAL_0_100_shuffle.txt',
        'LoadPEAL_1_100_shuffle.txt',
        'LoadPEAL_2_100_shuffle.txt',
        'LoadPEAL_3_100_shuffle.txt',
        'LoadPEAL_4_100_shuffle.txt'
        ]
for i in files:
    if i.endswith("txt"):
        print(i)