import os
dir_all_filename=os.listdir('C:/Users/luyih/PycharmProjects/matlabtopython/matlab2python/matlabcode')

for i in dir_all_filename:
    os.system("python matlab2python.py {0} -o {1}".format(i,i[:-1]+'py'))