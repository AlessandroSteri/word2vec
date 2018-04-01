import os
import matplotlib.pyplot as plt
import argparse
import numpy as np

def main():

    ### CLI args ###
    cmdLineParser = argparse.ArgumentParser()
    cmdLineParser.add_argument("id_exec", type=str, help="execution id")
    cmdLineParser.add_argument("other_id_exec", type=str, help="execution id")
    cmdLineParser.add_argument("dir", type=str, help="path/to/log/dir")
    cmdLineParser.add_argument("plot", type=str, help="what to plot: accuracy or loss")
    cmdLineArgs = cmdLineParser.parse_args()

    id_exec   = cmdLineArgs.id_exec
    other_id_exec   = cmdLineArgs.other_id_exec
    plot      = cmdLineArgs.plot
    directory = cmdLineArgs.dir

    to_plot = np.loadtxt(os.path.join(directory, plot, id_exec + '.csv'), delimiter=',')
    iterations = list(range(1, len(to_plot) + 1))
    other_to_plot = np.loadtxt(os.path.join(directory, plot, other_id_exec + '.csv'), delimiter=',')
    o_iterations = list(range(1, len(other_to_plot) + 1))
    # print(a)


    plt.figure(1)
    plt.plot(iterations, to_plot, o_iterations, other_to_plot)  # , time_opt, val_opt)
    plt.legend((id_exec, other_id_exec ))
    plt.title(id_exec + ' vs ' + other_id_exec)
    plt.ylabel(plot)
    plt.xlabel('iterations')
    plt.savefig(os.path.join(directory, id_exec + ".pdf") , bbox_inches='tight')
    plt.show()

# use to reload dict in domain
# # Load
# read_dictionary = np.load('my_file.npy').item()
# print(read_dictionary['hello']) # displays "world"


if __name__ == '__main__':
    main()

