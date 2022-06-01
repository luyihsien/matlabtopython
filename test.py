# import os
# dir_all_filename=os.listdir('C:/Users/luyih/PycharmProjects/matlabtopython/matlab2python/matlabcode')
#
# for i in dir_all_filename:
#     os.system("python matlab2python.py {0} -o {1}".format(i,i[:-1]+'py'))
import numpy as np
a=np.arange(1,3.5,0.5)
for i in a.reshape(-1):
    print(i)