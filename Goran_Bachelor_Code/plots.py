import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.use('TkAgg')  
import seaborn as sns
import matrixModification as mm
colors = ['#7dcdf5', '#d7f57d', '#e87d5f', '#5fe87f', 'sandybrown', 'lightcoral', 'grey', 'gold']
from matplotlib.colors import LogNorm



def avg_theta_by_d(N,df):
    df = df.query("N ==" +str(N)) 
    # print(df.head())
    # Calculate the average throughput for each heuristic for each value of d
    avg_throughput = df.groupby(['d', 'Alg'])['throughput'].mean().reset_index()

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot using seaborn
    sns.lineplot(data=avg_throughput, x='d', y='throughput', hue='Alg', marker='o')

    plt.title("Average Throughput by Heuristic for Different Values of d")
    plt.xlabel("Degree Constraint (d)")
    plt.ylabel("Average Throughput")
    plt.savefig("average_throughput_plot.png")
    plt.grid()
    plt.show()    
def avg_total_theta(df):
    avg_throughput = df.groupby(['Alg'])['throughput'].mean().reset_index()

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot using seaborn
    sns.barplot(data=avg_throughput, x='Alg', y='throughput')

    plt.title("Average Throughput by Heuristic")
    plt.xlabel("Heuristic")
    plt.ylabel("Average Throughput")

    plt.savefig("average_throughput_plot.png")
    plt.show()
def plot_by_matrix(matrixname, df):
    df = df[df['matrix'] == matrixname]

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot using seaborn
    sns.lineplot(data=df, x='d', y='throughput', hue='Alg', marker='o')

    plt.title("Throughput by Heuristic for Different Values of d")
    plt.xlabel("Degree Constraint (d)")
    plt.ylabel("Throughput")

    plt.savefig("average_throughput_plot.png")
    plt.show() 
def plot_2D_nparray(matrix):
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(matrix)
    ax.set_aspect('equal')
    # ax = sns.heatmap(renormalized_oblivious_matrix, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True),linewidths=2, cbar_kws={'ticks': ticks},vmin=minVal,vmax=maxVal)

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()
def filterDataframe(df, alt_rounding = False, sinkhorn = False, full_skews = False):
    if(alt_rounding):
        df = df[df['Alg'] != "Rounding"]
    else:
        df = df[df['Alg'] != "Alt_Rounding"]

    
    if(not full_skews):
        skews = ["skew-8-0.2", "skew-8-0.4", "skew-8-0.6", "skew-8-0.8", "skew-16-0.2", "skew-16-0.4", "skew-16-0.6", "skew-16-0.8"]
        df = df[~df['matrix'].isin(skews)]
    organicmatrices = ["data-parallelism","hybrid-parallelism","heatmap1","heatmap2","heatmap3", "topoopt"]
    if(sinkhorn):
       df = df[~df['matrix'].isin(organicmatrices)] 
    else:
        df = df[~df['matrix'].isin(['Sinkhorn_' + s for s in organicmatrices])]
    
    return df


if __name__ == "__main__":
    plotsdir = "/home/studium/Documents/Code/rdcn-throughput/Goran_Bachelor_Code/plots/"
    directory="/home/studium/Documents/Code/rdcn-throughput/Goran_Bachelor_Code/"
    matrixdir="/home/studium/Documents/Code/rdcn-throughput/matrices/"
    M = demandMatrix = np.loadtxt(matrixdir+"skew-8-0.8"+".mat", usecols=range(8))
    # Load the CSV file into a DataFrame
    df = pd.read_csv(directory +"outputFinal.csv", delim_whitespace=True, header=0)

    # Display first few rows to confirm loading

    # ax = sns.heatmap(renormalized_oblivious_matrix, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True),linewidths=2, cbar_kws={'ticks': ticks},vmin=minVal,vmax=maxVal)
    #Check example-vermillion file again
    df = df.drop(columns=df.columns[4])  # Drop irrelevant column
    df = filterDataframe(df, sinkhorn= True)
    print(df.head)
    # plot_2D_nparray(M)
    # plot_2D_nparray(mm.Sinkhorn_Knopp(M))
    # plot_by_matrix("data-parallelism", df)
    # avg_total_theta(df)
    # avg_theta_by_d(8, df)
    # avg_theta_by_d(16, df)