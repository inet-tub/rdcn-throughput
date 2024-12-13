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

def increasing_skew(N, d, df):
    # Define the set of matrices


    # Filter the DataFrame for the given N and d
    filtered_df = df[(df['N'] == N) & (df['d'] == d)]
    
    if(N == 16):
        filtered_df['matrix'] = pd.Categorical(filtered_df['matrix'], categories=skews16, ordered=True)
    elif(N==8):
        filtered_df['matrix'] = pd.Categorical(filtered_df['matrix'], categories=skews8, ordered=True)
    filtered_df = filtered_df.sort_values('matrix')

    # Set a modern style
    sns.set_theme(style="whitegrid", font_scale=1.5)
    
    # Create a custom color palette and marker styles
    custom_palette = sns.color_palette("husl", n_colors=len(filtered_df['Alg'].unique()))
    marker_styles = ['o', 's', 'D', '^', 'P', '*']  # Extend as needed

    # Initialize the plot with a larger size
    plt.figure(figsize=(12, 7))

    # Plot using seaborn
    sns.lineplot(
        data=filtered_df,
        x='matrix',
        y='throughput',
        hue='Alg',
        style='Alg',
        markers=marker_styles,
        dashes=False,
        palette=custom_palette,
        linewidth=2.5,
        markersize=10
    )

    # Add labels, title, and enhance gridlines
    plt.xlabel("Matrix")
    plt.ylabel("Throughput")
    # plt.title(f"Throughput Progression for N={N}, d={d}")
    plt.xticks(rotation=45)  # Rotate x-axis labels for readability
    plt.grid(color='grey', linestyle='-', linewidth=0.5, alpha=0.7)

    # Add a legend
    plt.legend(title="Algorithm", loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust layout for better appearance
    plt.tight_layout()
    plt.savefig("increasingskewsN="+str(N)+"d=" + str(d)+ ".png", dpi=300)

    # Show the plot
    plt.show()


def avg_theta_by_d(N, df):
    df = df.query("N == " + str(N))
    
    # Calculate the average throughput for each heuristic for each value of d
    avg_throughput = df.groupby(['d', 'Alg'])['throughput'].mean().reset_index()

    # Initialize the plot with a larger size
    plt.figure(figsize=(12, 7))

    # Set a modern style
    sns.set_theme(style="whitegrid", font_scale=1.5)
    
    # Use a custom color palette and marker styles
    custom_palette = sns.color_palette("husl", n_colors=len(avg_throughput['Alg'].unique()))
    marker_styles = ['o', 's', 'D', '^', 'P', '*']  # Add more if needed

    # Plot using seaborn
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

    # Custom title and labels
    # plt.title("Average Throughput by Heuristic for Different Values of d with N = "+ str(N), fontsize=16, weight='bold')
    plt.xlabel("Degree Constraint (d)", fontsize=22)
    plt.ylabel("Average Throughput", fontsize=22)
    
    # Customize legend
    plt.legend(title="Algorithm", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=12)
    
    # Save the figure with tight layout
    plt.tight_layout()
    plt.savefig("average_throughput_plot_stylishN"+str(N)+".png", dpi=300)
    
    # Show the plot
    plt.show()   
def min_theta_by_d(N, df):
    df = df.query("N == " + str(N))
    
    # Calculate the average throughput for each heuristic for each value of d
    avg_throughput = df.groupby(['d', 'Alg'])['throughput'].min().reset_index()

    # Initialize the plot with a larger size
    plt.figure(figsize=(12, 7))

    # Set a modern style
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    # Use a custom color palette and marker styles
    custom_palette = sns.color_palette("husl", n_colors=len(avg_throughput['Alg'].unique()))
    marker_styles = ['o', 's', 'D', '^', 'P', '*']  # Add more if needed

    # Plot using seaborn
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

    # Custom title and labels
    # plt.title("Minimum Throughput by Heuristic for Different Values of d with N = "+ str(N), fontsize=16, weight='bold')
    plt.xlabel("Degree Constraint (d)", fontsize=22)
    plt.ylabel("Minimum Throughput", fontsize=22)
    
    # Customize legend
    plt.legend(title="Algorithm", loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0., fontsize=12)
    
    # Save the figure with tight layout
    plt.tight_layout()
    plt.savefig("minimum_throughput_plot_stylishN"+str(N)+".png", dpi=300)
    
    # Show the plot
    plt.show()   

def avg_total_theta(df, N, d):
    df = df.query("N == "+str(N))
    df = df.query("d =="+str(d))
    # Initialize the plot with a larger vertical size
    plt.figure(figsize=(12, 8))

    # Set a modern theme
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    # Use a categorical color palette
    custom_palette = sns.color_palette("Set2", n_colors=len(df['Alg'].unique()))

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
    # plt.title("Throughput Distribution by Heuristic", fontsize=18, weight='bold', pad=20)
    plt.xlabel("Topology", fontsize=22)
    plt.ylabel("Throughput", fontsize=22)
    
    # Rotate x-axis labels for better visibility if necessary
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig("throughput_boxplot.png", dpi=300)
    
    # Show the plot
    plt.show()


def avg_total_theta_barplot(df):
    df = df.query("N == 8")
    # Calculate the average throughput for each heuristic
    avg_throughput = df.groupby(['Alg'])['throughput'].mean().reset_index()

    # Initialize the plot with a larger vertical size
    plt.figure(figsize=(12, 8))

    # Set a modern theme
    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    # Use a categorical color palette
    custom_palette = sns.color_palette("Set2", n_colors=len(avg_throughput['Alg'].unique()))

    # Plot using seaborn
    ax = sns.barplot(
        data=avg_throughput, 
        x='Alg', 
        y='throughput', 
        palette=custom_palette, 
        edgecolor='black'
    )

    # Annotate each bar with its value, with consistent offset
    for index, row in avg_throughput.iterrows():
        bar_height = row['throughput']
        ax.text(
            index, 
            bar_height + (bar_height * 0.02),  # Offset based on bar height
            f"{bar_height:.2f}", 
            ha='center', 
            va='bottom', 
            fontsize=11, 
            weight='bold'
        )

    # Custom title and labels
    plt.title("Average Throughput by Heuristic", fontsize=18, weight='bold', pad=20)
    plt.xlabel("Heuristic", fontsize=14)
    plt.ylabel("Average Throughput", fontsize=14)
    
    # Rotate x-axis labels for better visibility if necessary
    plt.xticks(rotation=45, ha='right', fontsize=12)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.savefig("average_throughput_barplot_fixed.png", dpi=300)
    
    # Show the plot
    plt.show()
def plot_by_matrix(matrixname, df):
    df = df[df['matrix'] == matrixname]

    # Initialize the plot
    plt.figure(figsize=(10, 6))

    # Plot using seaborn
    sns.lineplot(data=df, x='d', y='throughput', hue='Alg', marker='o')

    # plt.title("Throughput by Heuristic for Different Values of d")
    plt.xlabel("Degree Constraint (d)")
    plt.ylabel("Throughput")

    plt.savefig("average_throughput_plot.png")
    plt.show()
def plot_2D_nparray(matrix, logscale = True):
    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')

    # Apply a logarithmic normalization
    norm = LogNorm(vmin=np.min(matrix[matrix > 0]), vmax=np.max(matrix))
    im = plt.imshow(matrix, norm=norm, cmap='viridis')
    ax.set_aspect('equal')

    # Add a colorbar with the log scale
    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(im, orientation='vertical')
    plt.savefig("2Darray2.png")
    plt.show()
def filterDataframe(df, alt_rounding = False, sinkhorn = False, full_skews = False, onlyhalfskew = True, cicle= True, chord=True):
    if(alt_rounding):
        df = df[df['Alg'] != "Rounding"]
    else:
        df = df[df['Alg'] != "Alt_Rounding"]

    
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
    M = demandMatrix = np.loadtxt(matrixdir+"random-skewed-8"+".mat", usecols=range(8))
    fct.filtering(M)
    # Load the CSV file into a DataFrame
    df = pd.read_csv(directory +"outputFinal.csv", delim_whitespace=True, header=0)

    # Display first few rows to confirm loading

    # ax = sns.heatmap(renormalized_oblivious_matrix, cmap='GnBu',norm=colors.LogNorm(vmin=minVal,vmax=maxVal,clip=True),linewidths=2, cbar_kws={'ticks': ticks},vmin=minVal,vmax=maxVal)
    #Check example-vermillion file again
    df = df.drop(columns=df.columns[4])  # Drop irrelevant column
    df = filterDataframe(df, sinkhorn= True, full_skews= True)
    # # print(df.head)
    # plot_2D_nparray(M)
    # plot_2D_nparray(mm.Sinkhorn_Knopp(M))
    # plot_by_matrix("Sinkhorn_heatmap3", df)
    # avg_total_theta(df,16,10)
    # min_theta_by_d(16, df)
    # avg_theta_by_d(16, df)
    increasing_skew(16, 8, df)