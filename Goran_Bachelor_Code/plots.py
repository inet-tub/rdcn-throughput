import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
mpl.use('TkAgg')  
import seaborn as sns
import matrixModification as mm
colors = ['#7dcdf5', '#d7f57d', '#e87d5f', '#5fe87f', 'sandybrown', 'lightcoral', 'grey', 'gold']
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
import Throughput_as_Function as fct


skews8 = [
    "uniform-8",
    "skew-8-0.1",
    "skew-8-0.2",
    "skew-8-0.3",
    "skew-8-0.4",
    "skew-8-0.5",
    "skew-8-0.6",
    "skew-8-0.7",
    "skew-8-0.8",
    "skew-8-0.9",
    "permutation-8"
]
skews16 = [
    "uniform-16",
    "skew-16-0.1",
    "skew-16-0.2",
    "skew-16-0.3",
    "skew-16-0.4",
    "skew-16-0.5",
    "skew-16-0.6",
    "skew-16-0.7",
    "skew-16-0.8",
    "skew-16-0.9",
    "permutation-16"
]
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
alg_styles = {}
predefined_alg_order = ['Optimal', 'Floor', 'Rounding', 'RRG', 'Ring']
def initialize_styles(df):
    global alg_styles
    algs = [alg for alg in predefined_alg_order if alg in df['Alg'].unique()]
    custom_palette = sns.color_palette("husl", n_colors=len(algs))
    marker_styles = ['o', 's', 'D', '^', 'P', '*', 'v', 'X', 'h', '+']  # Extend if needed

    # Populate the dictionary with colors and markers
    alg_styles = {
        alg: {'color': custom_palette[i], 'marker': marker_styles[i % len(marker_styles)]}
        for i, alg in enumerate(algs)
    }
def increasing_skew(N, d, df):
    filtered_df = df[(df['N'] == N) & (df['d'] == d)]

    # Enforce matrix ordering
    if N == 16:
        filtered_df['matrix'] = pd.Categorical(filtered_df['matrix'], categories=skews16, ordered=True)
    elif N == 8:
        filtered_df['matrix'] = pd.Categorical(filtered_df['matrix'], categories=skews8, ordered=True)
    filtered_df = filtered_df.sort_values('matrix')

    # Enforce consistent Alg ordering
    filtered_df['Alg'] = pd.Categorical(filtered_df['Alg'], categories=predefined_alg_order, ordered=True)

    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", font_scale=1.5)
    
    # Use the predefined styles
    custom_palette = [alg_styles[alg]['color'] for alg in filtered_df['Alg'].cat.categories]
    marker_styles = [alg_styles[alg]['marker'] for alg in filtered_df['Alg'].cat.categories]

    ax =sns.lineplot(
        data=filtered_df,
        x='matrix',
        y='throughput',
        hue='Alg',
        style='Alg',
        markers=marker_styles,
        dashes=False,
        palette=custom_palette,
        linewidth=3,
        markersize=12
    )
    plt.xlabel("")
    plt.ylabel("Throughput")
    plt.xticks(rotation=45)
    plt.grid(color='grey', linestyle='-', linewidth=2, alpha=0.6)
    # ax.set_ylim(0.1,1.1)
    # ax.set_yscale("log")
    # ax.set_yticks([0.3, 0.5, 0.7, 0.9, 1])
    # ax.set_yticklabels(["0.3","0.5","0.7","0.9","1"])

    # Enforce consistent legend order
    plt.legend(title="Algorithm", loc="upper left", bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.savefig(f"increasingskewsN={N}d={d}.svg", format="svg", dpi=300)
    plt.show()
def avg_theta_by_d(N, df):
    df = df.query("N == " + str(N))
    avg_throughput = df.groupby(['d', 'Alg'])['throughput'].mean().reset_index()

    # Enforce consistent Alg ordering
    avg_throughput['Alg'] = pd.Categorical(avg_throughput['Alg'], categories=predefined_alg_order, ordered=True)

    # Print the averages
    print("[Alg] [d] [avg throughput]")
    for _, row in avg_throughput.iterrows():
        print(f"{row['Alg']} {row['d']} {row['throughput']:.3f}")
    
    plt.figure(figsize=(12, 7))
    sns.set_theme(style="whitegrid", font_scale=1.5)
    
    # Use the predefined styles
    custom_palette = [alg_styles[alg]['color'] for alg in avg_throughput['Alg'].cat.categories]
    marker_styles = [alg_styles[alg]['marker'] for alg in avg_throughput['Alg'].cat.categories]

    sns.lineplot(
        data=avg_throughput, 
        x='d', 
        y='throughput', 
        hue='Alg', 
        style='Alg', 
        markers=marker_styles, 
        dashes=False, 
        palette=custom_palette, 
        linewidth=2.5,
        markersize=10
    )
    plt.grid(color='grey', linestyle='-', linewidth=2, alpha=0.6)
    plt.xlabel("Degree Constraint (d)", fontsize=22)
    plt.ylabel("Average Throughput", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Enforce consistent legend order
    plt.legend(title="Algorithm", loc='upper left',  bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=16)

    plt.tight_layout()
    plt.savefig(f"average_throughput_plot_stylishN{N}.svg", format="svg", dpi=300)
    plt.show()
def min_theta_by_d(N, df):
    # Filter the DataFrame for the specified N
    df_filtered = df.query("N == @N")

    # Calculate the minimum throughput for each heuristic for each value of d
    min_throughput = df_filtered.groupby(['d', 'Alg'])['throughput'].min().reset_index()

    # Merge back with the original dataframe to get the corresponding 'matrix' value
    merged_df = pd.merge(min_throughput, df_filtered, on=['d', 'Alg', 'throughput'], how='left')

    # Print the details for each minimum throughput point
    for _, row in merged_df.iterrows():
        print(f"{row['Alg']} {row['d']} {row['throughput']} {row['matrix']}")

    # Enforce consistent Alg ordering
    min_throughput['Alg'] = pd.Categorical(min_throughput['Alg'], categories=predefined_alg_order, ordered=True)

    # Initialize the plot with a larger size
    plt.figure(figsize=(12, 7))

    # Set a modern style
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    # Use the predefined styles
    custom_palette = [alg_styles[alg]['color'] for alg in min_throughput['Alg'].cat.categories]
    marker_styles = [alg_styles[alg]['marker'] for alg in min_throughput['Alg'].cat.categories]

    # Plot using seaborn
    sns.lineplot(
        data=min_throughput, 
        x='d', 
        y='throughput', 
        hue='Alg', 
        style='Alg', 
        markers=marker_styles, 
        dashes=False, 
        palette=custom_palette, 
        linewidth=2.5,
        markersize=10
    )
    plt.grid(color='grey', linestyle='-', linewidth=2, alpha=0.6)
    # Custom title and labels
    plt.xlabel("Degree Constraint (d)", fontsize=22)
    plt.ylabel("Minimum Throughput", fontsize=22)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # Customize legend with consistent ordering
    plt.legend(title="Algorithm", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=16)

    # Save the figure with tight layout
    plt.tight_layout()
    plt.savefig(f"minimum_throughput_plot_stylishN{N}.svg", format="svg", dpi=300)

    # Show the plot
    plt.show()
def avg_total_SH(df):
    # Enforce consistent Alg ordering
    df['Alg'] = pd.Categorical(df['Alg'], categories=predefined_alg_order, ordered=True)
    
    # Compute mean throughput per algorithm
    mean_throughput = df.groupby('Alg')['SH'].mean()

    # Print the mean throughput for each algorithm
    print("Mean SH share per Algorithm:")
    for alg, mean_value in mean_throughput.items():
        print(f"{alg}: {mean_value:.3f}")

    # Initialize the plot with a larger size
    plt.figure(figsize=(12, 8))

    # Set a modern style
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    # Use predefined colors for the algorithms
    custom_palette = [alg_styles[alg]['color'] for alg in df['Alg'].cat.categories]

    # Plot using seaborn boxplot
    ax = sns.boxplot(
        data=df,
        x='Alg',
        y='SH',
        palette=custom_palette,
        linewidth=1.5,
        showmeans=True,  # Show the mean
        meanline=True,  # Mean represented as a line
        meanprops={
            "color": "red", 
            "linestyle": "--", 
            "linewidth": 2
        },
        boxprops={"edgecolor": "black"},  # Outline the box
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "blue", "linewidth": 2}  # Highlight median
    )

    # Custom title and labels
    plt.xlabel("")  # Remove x-axis label
    plt.ylabel("SH", fontsize=22)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(f"SH_boxplot_allNs.svg", format="svg", dpi=300)
    
    # Show the plot
    plt.show()
def avg_total_theta(df):
    # Enforce consistent Alg ordering
    df['Alg'] = pd.Categorical(df['Alg'], categories=predefined_alg_order, ordered=True)
    
    # Compute mean throughput per algorithm
    mean_throughput = df.groupby('Alg')['throughput'].mean()

    # Print the mean throughput for each algorithm
    print("Mean Throughput per Algorithm:")
    for alg, mean_value in mean_throughput.items():
        print(f"{alg}: {mean_value:.3f}")

    # Initialize the plot with a larger size
    plt.figure(figsize=(12, 8))

    # Set a modern style
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    # Use predefined colors for the algorithms
    custom_palette = [alg_styles[alg]['color'] for alg in df['Alg'].cat.categories]

    # Plot using seaborn boxplot
    ax = sns.boxplot(
        data=df,
        x='Alg',
        y='throughput',
        palette=custom_palette,
        linewidth=1.5,
        showmeans=True,  # Show the mean
        meanline=True,  # Mean represented as a line
        meanprops={
            "color": "red", 
            "linestyle": "--", 
            "linewidth": 2
        },
        boxprops={"edgecolor": "black"},  # Outline the box
        whiskerprops={"color": "black"},
        capprops={"color": "black"},
        medianprops={"color": "blue", "linewidth": 2}  # Highlight median
    )

    # Custom title and labels
    plt.xlabel("")  # Remove x-axis label
    plt.ylabel("Throughput", fontsize=22)
    
    # Rotate x-axis labels for better visibility
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig(f"throughput_boxplot_allNs.svg", format="svg", dpi=300)
    
    # Show the plot
    plt.show()

def plot_2D_nparray(matrix, logscale=True, vmin=1e-3, vmax=1, name= "2Darray"):  # vmin and vmax added
    """
    Plots a 2D numpy array with a fixed color scale.

    Parameters:
        matrix (2D np.array): The array to plot.
        logscale (bool): Whether to use log scaling for the color normalization.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
    """
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    # ax.set_title('colorMap')

    # Apply a logarithmic normalization with fixed vmin and vmax
    if logscale:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None  # Use linear scaling

    # Plot the matrix
    im = plt.imshow(matrix, norm=norm, cmap='viridis')
    ax.set_aspect('equal')

    # Add a colorbar
    plt.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04)
    
    # Save and display the figure
    plt.savefig(name +".svg", format="svg", bbox_inches='tight')
    plt.show()
def plot_2D_nparray_with_labels(matrix, vmin=1e-3, vmax=6, name="2Darray"):
    """
    Plots a 2D numpy array using a logarithmic colormap, labels non-zero integer values,
    and removes the y-axis labels.

    Parameters:
        matrix (2D np.array): The array to plot.
        vmin (float): Minimum value for the color scale.
        vmax (float): Maximum value for the color scale.
        name (str): Filename for saving the plot.
    """
    fig, ax = plt.subplots(figsize=(6, 3.2))

    # Use LogNorm for the colormap
    norm = LogNorm(vmin=vmin, vmax=vmax)
    im = ax.imshow(matrix, norm=norm, cmap='viridis')
    ax.set_aspect('equal')

    # Add labels for non-zero integers
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if value != 0 and int(value) == value:  # Label only non-zero integers
                ax.text(j, i, f"{int(value)}", color="black", ha='center', va='center', fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, orientation='vertical', fraction=0.046, pad=0.04)
    # cbar.set_ticks([0.001, 1, 2, 3, 4, 5, 6])  # Set positions for integers on the log scale
    # cbar.set_ticklabels([0, 1, 2, 3, 4, 5, 6])  # Set corresponding integer labels

    # Remove y-axis labels
    # ax.yaxis.set_visible(False)

    # Save and display the figure
    plt.savefig(name + ".svg", format="svg", bbox_inches='tight')
    plt.show()

def filterDataframe(df, alt_rounding = False, sinkhorn = False, full_skews = False, onlyhalfskew = True, cicle= True, chord=False):
    df = df.replace(to_replace='Circle', value='Ring', regex=True)
    if(alt_rounding):
        df = df[df['Alg'] != "Rounding"]
    else:
        df = df[df['Alg'] != "Alt_Rounding"]
    if(not chord):
        df = df[df['Alg'] != "Chord"]
    
    if(not full_skews):
        skews = ["skew-8-0.2", "skew-8-0.4", "skew-8-0.6", "skew-8-0.8", "skew-16-0.2", "skew-16-0.4", "skew-16-0.6", "skew-16-0.8"]
        df = df[~df['matrix'].isin(skews)]
        if(onlyhalfskew):#Filter everything but skew-16-0.5
            skews = ["skew-8-0.1", "skew-8-0.3", "skew-8-0.7", "skew-8-0.9", "skew-16-0.1", "skew-16-0.3", "skew-16-0.7", "skew-16-0.9"]
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
    # Mname = "hybrid-parallelism"
    # M = demandMatrix = np.loadtxt(matrixdir+Mname+".mat", usecols=range(16))


    # dB  = np.loadtxt("Rounded.txt", usecols=range(8))
    # dF  = np.loadtxt("RoundedRRG.txt", usecols=range(8))
    # dRRG = np.loadtxt("RRGadded.txt", usecols=range(8))
    # dGCM = np.loadtxt("GCMadded.txt", usecols=range(8))
    # plot_2D_nparray(dB, name="demandBefore", vmax=6)
    # plot_2D_nparray_with_labels(dB, name="Rounded")
    # plot_2D_nparray_with_labels(dF, name="RoundedRRG")
    # plot_2D_nparray_with_labels(dGCM, name="demandGCM")

    
    # Load the CSV file into a DataFrame
    # df = pd.read_csv(directory +"lastOutput.csv", delim_whitespace=True, header=0)
    df = pd.read_csv(directory +"output2NoFloor.csv", delim_whitespace=True, header=0)


    # Display first few rows to confirm loading

    # # ax = sns.heatmap(renormalized_oblivious_matrix, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True),linewidths=2, cbar_kws={'ticks': ticks},vmin=minVal,vmax=maxVal)
    # #Check example-vermillion file again
    # df = df.drop(columns=df.columns[3])
    # df = df.drop(columns=df.columns[4])  # Drop irrelevant column

    df = filterDataframe(df, sinkhorn= True, full_skews= False)
    # print(df['throughput'])
    initialize_styles(df)

    # print(df.head)
    # fct.filtering(M)
    # plot_2D_nparray(M, name=Mname)
    
    # plot_2D_nparray(mm.Sinkhorn_Knopp(M), name=Mname + "-cleaned")
    # plot_by_matrix("Sinkhorn_heatmap3", df)
    avg_total_theta(df)
    # min_theta_by_d(8, df)
    # min_theta_by_d(16, df)
    # avg_theta_by_d(8, df)
    # avg_theta_by_d(16, df)

    # increasing_skew(8, 3, df) 
    # increasing_skew(8, 5, df) 
    # increasing_skew(8, 7, df) 
    # increasing_skew(16, 6, df) 
    # increasing_skew(16, 10, df) 
    # increasing_skew(16, 14, df) 