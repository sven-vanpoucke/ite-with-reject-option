import matplotlib.pyplot as plt

def onelinegraph(x, x_label, y, y_label, color, title, folder, xmin=0, xmax=0, ymin=0, ymax=0):
    # Graph with reject rate and RMSE of Accepted Samples
    plt.plot(x, y, color=color, label=y_label)
    plt.xlabel(x_label)
    plt.title(title)
    if xmin != xmax:
        plt.xlim(xmin, xmax)  # Set x-axis range from 0 to 6
    if ymin != ymax:
        plt.ylim(ymin, ymax)  # Set y-axis range from 0 to 25
    plt.legend()
    plt.savefig(folder)
    plt.close()
    plt.cla()

def onelinechangegraph(x, x_label, y, y_label, color, title, folder, xmin=0, xmax=0, ymin=0, ymax=0):
    # Graph with reject rate and RMSE of Accepted Samples
    plt.plot(x, y, color=color, label=y_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.title(title)
    plt.ylim(-9, 3)  # Set x-axis range from 0 to 6
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1)
    # plt.xlim(xmin, xmax)  # Set y-axis range from 0 to 25
    # plt.legend()
    plt.savefig(folder)
    plt.close()
    plt.cla()


def twolinegraph(x, x_label, y, y_label, color, y2, y2_label, color2, title, folder):
    # Graph with reject rate and RMSE of Accepted Samples
    plt.plot(x, y, color=color, label=y_label)
    plt.plot(x, y2, color=color2, label=y2_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.legend()
    plt.savefig(folder)
    plt.close()
    plt.cla()

def histogram(values, xlabel, ylabel, title, folder, lowest_rejected_value, mean, plusstd, plus2std):
    plt.hist(values)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.axvline(mean, color='blue', linestyle='dashed', linewidth=2, label='Mean')
    plt.axvline(lowest_rejected_value, color='red', linestyle='dashed', linewidth=2, label='Lowest R')
    plt.axvline(plusstd, color='green', linestyle='dashed', linewidth=2, label='Mean + 1 Std Dev')
    # plt.axvline(plus2std, color='green', linestyle='dashed', linewidth=2, label='Mean + 2 Std Dev')
    plt.legend()
    plt.title(title)
    plt.savefig(folder)
    plt.close()
    plt.cla()


