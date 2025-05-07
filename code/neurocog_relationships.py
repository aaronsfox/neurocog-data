# -*- coding: utf-8 -*-
"""

@author:

    Aaron Fox
    Centre for Sport Research
    Deakin University
    aaron.f@deakin.edu.au
    
    This script compiles the neurocognitive test data together to explore
    performance relationships between the tests.

"""

# =========================================================================
# Import packages
# =========================================================================

import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from matplotlib import rcParams
import re
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# =========================================================================
# Set-up
# =========================================================================

# General settings
# -------------------------------------------------------------------------

# Set matplotlib parameters
matplotlib.use('TkAgg')
plt.ion()
rcParams['font.sans-serif'] = 'Arial'
rcParams['font.weight'] = 'bold'
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 16
rcParams['axes.linewidth'] = 1.5
rcParams['axes.labelweight'] = 'bold'
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['legend.fontsize'] = 10
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['legend.framealpha'] = 0.0
rcParams['savefig.dpi'] = 300
rcParams['savefig.format'] = 'pdf'

# Create colour map for correlation matrix
corrColMap = sns.blend_palette(['#156082','#DCEAF7','#156082'], as_cmap=True)

#Set test order and labels for plotting
testPlotOrder = ['SRT', '4Choice', 'SportRT', '1B', '2B', '3B', 'OCL']
testPlotNames = ['Simple RT', '4-Choice RT', 'Sport RT', '1-Back Test', '2-Back Test', '3-Back Test', '1-Card Learning']

# Analysis settings
# -------------------------------------------------------------------------

# Set desired alpha level for correlations statistics
alpha = 0.05

# Participant settings
# -------------------------------------------------------------------------

#Set participant dictionary with useful info
participantInfo = {
    'P01': {'age': 22, 'height': 1.72, 'mass': np.nan},
    'P02': {'age': 21, 'height': 1.70, 'mass': 59.5},
    'P03': {'age': 28, 'height': 1.618, 'mass': 57.55},
    'P04': {'age': 29, 'height': 1.895, 'mass': 89.1},
    'P05': {'age': 21, 'height': 1.674, 'mass': 62.25},
    'P08': {'age': 21, 'height': 1.646, 'mass': 67.25},
    'P09': {'age': 19, 'height': 1.67, 'mass': 69.0},
    'P10': {'age': 30, 'height': 1.62, 'mass': 58.65},
    'P11': {'age': 27, 'height': 1.76, 'mass': 79.0},
    'P12': {'age': 31, 'height': 1.71, 'mass': 59.95},
    'P13': {'age': 35, 'height': 1.66, 'mass': 57.8},
    'P14': {'age': 39, 'height': 1.64, 'mass': 65.3},
    'P15': {'age': 21, 'height': 1.62, 'mass': 67.0},
    'P16': {'age': 21, 'height': 1.70, 'mass': 58.1},
    'P17': {'age': 24, 'height': 1.69, 'mass': 63.05},
    'P18': {'age': 18, 'height': 1.75, 'mass': 69.05},
    'P19': {'age': 24, 'height': 1.69, 'mass': 69.65},
    'P20': {'age': 18, 'height': 1.75, 'mass': 69.05},
    'P21': {'age': 25, 'height': 1.66, 'mass': 75.1},
    'P22': {'age': 23, 'height': 1.68, 'mass': 62.5},
}

# =========================================================================
# Define functions
# =========================================================================

# Read in the neurocognitive data from listed participants
# -------------------------------------------------------------------------
def read_neurocog_data(participant_list):

    """
    :param participant_list: list of participants to extract data for
    :return: dataframe of participant neurocog data

    """

    # Set-up list to store dataframes in for later concatenation
    neurocogDataList = []

    # Loop through participants
    for participantId in participant_list:

        # Load in participant neurocognitive results
        nDf = pd.read_csv(os.path.join('..', 'data', participantId, f'{participantId}_neurocog_data.csv'))

        # Add in participant Id column
        nDf['participantId'] = [participantId] * len(nDf)

        # Append to list
        neurocogDataList.append(nDf)

    # Concatenate all data to one dataframe
    neurocog_data = pd.concat(neurocogDataList)

    # Return dataframe
    return neurocog_data


# Create descriptives figure for provided participant data
# -------------------------------------------------------------------------
def generate_boxplot(participant_data):

    """
    :param participant_data: dataframe of participant neurocog data
    :return: figure and axes handles

    Function creates a subplot figure with the tests as the x-axis and the results on
    the y-axis. One subplot shows reaction time (on correct responses) for tests, and
    the other subplot shows the percentage accuracy on the tests.

    """

    # Set test order and labels for plotting
    testPlotOrder = ['SRT', '4Choice', '1B', '2B', '3B', 'OCL', 'SportRT']
    testPlotNames = ['Simple RT', '4-Choice RT', '1-Back Test', '2-Back Test', '3-Back Test', '1-Card Learning', 'Sport RT']

    # Set the colour palette from seaborn
    plotColPal = sns.color_palette(palette='colorblind', n_colors=len(testPlotOrder))

    # Set markers for points
    plotMarkers = ['o', '^', 's', 'p', 'd', 'v', 'H']

    # Create figure and axes
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15, 7),
                           sharex=False, sharey=False)

    # Boxplot for reaction time measures
    # -------------------------------------------------------------------------

    # Create reaction time boxplot
    bp1 = sns.boxplot(participant_data.loc[participant_data['variableName'] == 'avgCorrectReactionTime'],
                      x='testName', y='result',
                      order=testPlotOrder, boxprops=dict(facecolor='white', edgecolor='black'), linewidth=1.5,
                      width = 0.4, fliersize=0, zorder=2,
                      ax=ax[0])

    # Adjust the colouring of the boxplot lines
    for patch, color in zip(bp1.patches, plotColPal):
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_facecolor('white')

    #  Update lines manually
    # The order in ax.lines is:
    # [whisker, whisker, cap, cap, median, whisker, whisker, cap, cap, median, ...]
    lines = bp1.lines
    n_boxes = len(bp1.patches)
    for i in range(n_boxes):
        color = plotColPal[i]
        # Find the lines for this box
        whisker1 = lines[i * 6 + 0]
        whisker2 = lines[i * 6 + 1]
        cap1 = lines[i * 6 + 2]
        cap2 = lines[i * 6 + 3]
        median = lines[i * 6 + 4]
        # fliers would be lines[i*6 + 5], but ignore them
        whisker1.set_color(color)
        whisker2.set_color(color)
        cap1.set_color(color)
        cap2.set_color(color)
        median.set_color(color)
        median.set_linewidth(1.5)

    # Add the strip plot for points
    # Hacky loop way, but allows greater flexibility in aspects when plotting
    for testPlot in testPlotOrder:
        sp = sns.stripplot(x=testPlotOrder.index(testPlot),
                           y=participant_data.loc[(participant_data['testName'] == testPlot) &
                                                  (participant_data['variableName'] == 'avgCorrectReactionTime'),
                           ]['result'].to_numpy(),
                           color=plotColPal[testPlotOrder.index(testPlot)],
                           marker=plotMarkers[testPlotOrder.index(testPlot)],
                           size=6, alpha=0.80,
                           jitter=True, dodge=False,
                           native_scale=True, zorder=4,
                           ax=ax[0])

    # Set axes title
    ax[0].set_title('Avg. Correct Reaction Time', fontsize=14, fontweight='bold')

    # Remove x-label
    ax[0].set_xlabel('')

    # Set y-label
    ax[0].set_ylabel('Reaction Time (ms)', fontsize=12, fontweight='bold', labelpad=15)

    # Configure x-tick labels
    ax[0].set_xticks(ax[0].get_xticks())
    ax[0].set_xticklabels(testPlotNames, rotation=45, ha='right')

    # Add zero line
    ax[0].axhline(0, c='dimgrey', lw=0.5, ls='--', zorder=2)

    # Boxplot for accuracy measures
    # -------------------------------------------------------------------------

    # Create accuracy boxplot
    bp2 = sns.boxplot(participant_data.loc[participant_data['variableName'] == 'responseAccuracy'],
                      x='testName', y='result',
                      order=testPlotOrder, boxprops=dict(facecolor='white', edgecolor='black'), linewidth=1.5,
                      width=0.4, fliersize=0, zorder=2,
                      ax=ax[1])

    # Adjust the colouring of the boxplot lines
    for patch, color in zip(bp2.patches, plotColPal):
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
        patch.set_facecolor('white')

    #  Update lines manually
    # The order in ax.lines is:
    # [whisker, whisker, cap, cap, median, whisker, whisker, cap, cap, median, ...]
    lines = bp2.lines
    n_boxes = len(bp2.patches)
    for i in range(n_boxes):
        color = plotColPal[i]
        # Find the lines for this box
        whisker1 = lines[i * 6 + 0]
        whisker2 = lines[i * 6 + 1]
        cap1 = lines[i * 6 + 2]
        cap2 = lines[i * 6 + 3]
        median = lines[i * 6 + 4]
        # fliers would be lines[i*6 + 5], but ignore them
        whisker1.set_color(color)
        whisker2.set_color(color)
        cap1.set_color(color)
        cap2.set_color(color)
        median.set_color(color)
        median.set_linewidth(1.5)

    # Add the strip plot for points
    # Hacky loop way, but allows greater flexibility in aspects when plotting
    for testPlot in testPlotOrder:
        sp = sns.stripplot(x=testPlotOrder.index(testPlot),
                           y=participant_data.loc[(participant_data['testName'] == testPlot) &
                                                  (participant_data['variableName'] == 'responseAccuracy'),
                           ]['result'].to_numpy(),
                           color=plotColPal[testPlotOrder.index(testPlot)],
                           marker=plotMarkers[testPlotOrder.index(testPlot)],
                           size=6, alpha=0.80,
                           jitter=True, dodge=False,
                           native_scale=True, zorder=4,
                           ax=ax[1])

    # Set axes title
    ax[1].set_title('Response Accuracy', fontsize=14, fontweight='bold')

    # Remove x-label
    ax[1].set_xlabel('')

    # Set y-label
    ax[1].set_ylabel('Proportion of Correct Responses (%)', fontsize=12, fontweight='bold', labelpad=15)

    # Configure x-tick labels
    ax[1].set_xticks(ax[1].get_xticks())
    ax[1].set_xticklabels(testPlotNames, rotation=45, ha='right')

    # Set y-axes to start just below zero
    ax[1].set_ylim([0, 105])

    # Set corresponding appropriate y-ticks
    ax[1].set_yticks([0, 20, 40, 60, 80, 100])

    # Finalise plot
    # -------------------------------------------------------------------------

    # Set tight layout option
    plt.tight_layout()

    # Return figure and axes handles
    return fig, ax


# Analyse neurocognitive tests vs. sport-specific task
# -------------------------------------------------------------------------
def relationships_sportRT(participant_data, variable_name, outcome_label, plot_corrMat=False):

    """
    Function analyses correlations between neurocognitive tests to SportRT test for provided variable name

    :param participant_data (pandas dataframe): compiled dataframe of participant neurocognitive performance data
    :param variable_name (string): variable name to analyse relationships against
    :param outcome_label (string): descriptor of the outcome variable used to add to plots (e.g. 'Response Time (ms)')
    :param plot_corrMat (boolean): whether to plot a heatmap of correlation matrix (not necessary but can be useful to review, default = False)

    """

    # Pivot the dataframe wider
    # For response accuracy include the arcsin transformation here & drop SRT as it isn't relevant
    if variable_name == 'responseAccuracy':
        data_wide = (participant_data.loc[(participant_data['variableName'] == variable_name) &
                                          (participant_data['testName'] != 'SRT'),].pivot(
            index='participantId', columns='testName', values='result') / 100).apply(np.arcsin)
    else:
        data_wide = participant_data.loc[participant_data['variableName'] == variable_name,].pivot(
            index='participantId', columns='testName', values='result')

    # Create the correlation matrix
    corr_mat = data_wide.corr()

    # Calculate r and p-values separately
    corr_r = data_wide.corr(method=lambda x, y: pearsonr(x, y)[0]) - np.eye(*corr_mat.shape)
    corr_p = data_wide.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*corr_mat.shape)

    # Plotting the correlation matrix can help with reviewing the results
    # An option is given earlier whether to plot the correlation matrix as it isn't entirely necessary
    if plot_corrMat:

        # Get the mask to plot half the matrix
        mask_matrix = np.triu(np.ones_like(corr_mat))

        # Create correlation heatmap to view
        corr_mat_heat = sns.heatmap(corr_mat,
                                    cmap=corrColMap,
                                    vmin=-0.7, vmax=0.7,
                                    mask=mask_matrix)

        # Add figure labelling
        corr_mat_heat.set_title(' '.join(re.findall(r'[A-Z]?[a-z]+', variable_name)).title(), fontweight='bold')

        # Add correlations and outlines for statistical significance

        # Loop through dataframe shape
        for iiRow in range(corr_p.shape[0]):
            for iiCol in range(corr_p.shape[1]):

                # Only examine lower triangle
                if iiRow > iiCol:

                    # Get r and p-value
                    rVal = corr_r.to_numpy()[iiRow, iiCol]
                    pVal = corr_p.to_numpy()[iiRow, iiCol]

                    # Add correlation text
                    plt.gca().text(iiCol + 0.5, iiRow + 0.5, '{0:.2f}'.format(rVal),
                                   ha='center', va='center',
                                   color='white', fontsize=12, fontweight='bold')

                    # Outline if p < alpha
                    if pVal < alpha:
                        # Add rectangle patch
                        corr_mat_heat.add_patch(Rectangle((iiCol, iiRow), 1, 1,
                                                          ec='black', fc='none', lw=2.0,
                                                          clip_on=False, zorder=3))

        # Remove useless x and y labels
        plt.gca().set_xlabel('')
        plt.gca().set_ylabel('')

        # Get appropriate tick labels
        # Map renaming to dictionary
        tickRenamer = {testPlotOrder[ii]: testPlotNames[ii] for ii in range(len(testPlotOrder))}
        # Get the labels
        labels = plt.gca().get_xticklabels()
        # Create new labels
        newLabels = []
        for label in labels:
            newLabels.append(tickRenamer[label.get_text()])

        # Set the tick labels
        plt.gca().set_xticklabels(newLabels, rotation=45, ha='right', va='top')
        plt.gca().set_yticklabels(newLabels, rotation=45, ha='right', va='top')

        # Drop the unnecessary 1-back test from the y-axis
        plt.gca().set_ylim([plt.gca().get_ylim()[0], plt.gca().get_ylim()[1] + 0.9])

        # Drop the unnecessary SportRT test from the x-axis
        plt.gca().set_xlim([plt.gca().get_xlim()[0], plt.gca().get_xlim()[1] - 0.9])

        # Remove the color bar
        plt.gcf().delaxes(plt.gcf().axes[1])

        # Set axis position
        plt.gca().set_position([0.175, 0.19, 0.75, 0.75])

    # Create scatter plot of tests vs. sport-specific task

    # Set correlation plot order
    corrPlotOrder = ['SRT', '4Choice', '1B', '2B', '3B', 'OCL']
    corrPlotNames = ['Simple RT', '4-Choice RT', '1-Back Test', '2-Back Test', '3-Back Test', '1-Card Learning']

    # Create figure and axis
    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(8, 11),
                           sharex=False, sharey=False)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95,
                        wspace=0.35, hspace=0.4)

    # Loop through correlations and plot
    for test in corrPlotOrder:

        # If variable is response accuracy and test is SRT don't plot as accuracy is 100% for everyone
        if variable_name == 'responseAccuracy' and test == 'SRT':

            ax.flatten()[corrPlotOrder.index(test)].axis('off')

        else:

            # Create the scatter plot
            sns.regplot(data=data_wide, x='SportRT', y=test,
                        marker='o', color='black',
                        scatter_kws={'s': 10},
                        line_kws={'color': 'black', 'lw': 1.0, 'ls': '--'},
                        ax=ax.flatten()[corrPlotOrder.index(test)])

            # Set x and y labels
            ax.flatten()[corrPlotOrder.index(test)].set_ylabel(corrPlotNames[corrPlotOrder.index(test)] + f' {outcome_label}',
                                                               fontsize=10, fontweight='bold', labelpad=10)
            ax.flatten()[corrPlotOrder.index(test)].set_xlabel(f'Sport-Specific Test {outcome_label}',
                                                               fontsize=10, fontweight='bold', labelpad=10)

            # Add test title
            ax.flatten()[corrPlotOrder.index(test)].set_title(corrPlotNames[corrPlotOrder.index(test)],
                                                              fontsize=12, fontweight='bold', pad=5)

            # Add correlation statistical info
            # Positioning for reaction time
            if variable_name == 'avgCorrectReactionTime':
                if test in ['SRT', '4Choice', '1B', 'OCL']:
                    # r-value
                    ax.flatten()[corrPlotOrder.index(test)].text(
                        ax.flatten()[corrPlotOrder.index(test)].get_xlim()[0] + (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_xlim())[0] * 0.05),
                        ax.flatten()[corrPlotOrder.index(test)].get_ylim()[-1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_ylim())[0] * 0.05),
                        f'r = {"{0:.3f}".format(corr_r[test]["SportRT"])}',
                        ha='left', va='center', color='red', fontsize=10, fontweight='normal'
                    )
                    # p-value
                    ax.flatten()[corrPlotOrder.index(test)].text(
                        ax.flatten()[corrPlotOrder.index(test)].get_xlim()[0] + (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_xlim())[0] * 0.05),
                        ax.flatten()[corrPlotOrder.index(test)].get_ylim()[-1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_ylim())[0] * 0.12),
                        f'p = {"{0:.3f}".format(corr_p[test]["SportRT"])}',
                        ha='left', va='center', color='red', fontsize=10, fontweight='normal'
                    )
                else:
                    # r-value
                    ax.flatten()[corrPlotOrder.index(test)].text(
                        ax.flatten()[corrPlotOrder.index(test)].get_xlim()[1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_xlim())[0] * 0.05),
                        ax.flatten()[corrPlotOrder.index(test)].get_ylim()[-1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_ylim())[0] * 0.05),
                        f'r = {"{0:.3f}".format(corr_r[test]["SportRT"])}',
                        ha='right', va='center', color='red', fontsize=10, fontweight='normal'
                    )
                    # p-value
                    ax.flatten()[corrPlotOrder.index(test)].text(
                        ax.flatten()[corrPlotOrder.index(test)].get_xlim()[1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_xlim())[0] * 0.05),
                        ax.flatten()[corrPlotOrder.index(test)].get_ylim()[-1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_ylim())[0] * 0.12),
                        f'p = {"{0:.3f}".format(corr_p[test]["SportRT"])}',
                        ha='right', va='center', color='red', fontsize=10, fontweight='normal'
                    )
            # Positioning for response accuracy
            if variable_name == 'responseAccuracy':
                if test not in ['3B']:
                    # r-value
                    ax.flatten()[corrPlotOrder.index(test)].text(
                        ax.flatten()[corrPlotOrder.index(test)].get_xlim()[1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_xlim())[0] * 0.05),
                        ax.flatten()[corrPlotOrder.index(test)].get_ylim()[0] + (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_ylim())[0] * 0.12),
                        f'r = {"{0:.3f}".format(corr_r[test]["SportRT"])}',
                        ha='right', va='center', color='red', fontsize=10, fontweight='normal'
                    )
                    # p-value
                    ax.flatten()[corrPlotOrder.index(test)].text(
                        ax.flatten()[corrPlotOrder.index(test)].get_xlim()[1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_xlim())[0] * 0.05),
                        ax.flatten()[corrPlotOrder.index(test)].get_ylim()[0] + (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_ylim())[0] * 0.05),
                        f'p = {"{0:.3f}".format(corr_p[test]["SportRT"])}',
                        ha='right', va='center', color='red', fontsize=10, fontweight='normal'
                    )
                else:
                    # r-value
                    ax.flatten()[corrPlotOrder.index(test)].text(
                        ax.flatten()[corrPlotOrder.index(test)].get_xlim()[1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_xlim())[0] * 0.05),
                        ax.flatten()[corrPlotOrder.index(test)].get_ylim()[-1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_ylim())[0] * 0.05),
                        f'r = {"{0:.3f}".format(corr_r[test]["SportRT"])}',
                        ha='right', va='center', color='red', fontsize=10, fontweight='normal'
                    )
                    # p-value
                    ax.flatten()[corrPlotOrder.index(test)].text(
                        ax.flatten()[corrPlotOrder.index(test)].get_xlim()[1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_xlim())[0] * 0.05),
                        ax.flatten()[corrPlotOrder.index(test)].get_ylim()[-1] - (np.diff(ax.flatten()[corrPlotOrder.index(test)].get_ylim())[0] * 0.12),
                        f'p = {"{0:.3f}".format(corr_p[test]["SportRT"])}',
                        ha='right', va='center', color='red', fontsize=10, fontweight='normal'
                    )

    # Return outputs
    return corr_r, corr_p, fig, ax


# =========================================================================
# Run analysis
# =========================================================================

if __name__ == '__main__':

    # =========================================================================
    # Compile participant neurocog data
    # =========================================================================

    # Generate list of non-concussed participants
    participants = [participant for participant in participantInfo.keys()]

    # Get neurocognitive data for non-concussed participants
    nc_data = read_neurocog_data(participants)

    # Generate descriptives figure for non-concussed participant data
    fig, ax = generate_boxplot(nc_data)

    # Save figure
    fig.savefig(os.path.join('..','results', 'figures', 'boxPlot_descriptive_neurocog-results.png'),
                format='png', dpi=600)

    # Close figure
    plt.close('all')

    # =========================================================================
    # Analyse relationships between neurocognitive performance tests
    # =========================================================================

    # Average correct reaction time
    # -------------------------------------------------------------------------

    # Run analysis function and return outputs
    corr_r_responseTime, corr_p_responseTime, fig_responseTime, ax_responseTime = relationships_sportRT(
        nc_data, 'avgCorrectReactionTime', 'Response Time (ms)')

    # Save figure
    fig_responseTime.savefig(os.path.join('..','results', 'figures', 'scatter_reaction-time_SportRT.png'),
                             format='png', dpi=600)

    # Close figure
    plt.close(fig_responseTime)

    # Response accuracy
    # -------------------------------------------------------------------------

    # Run analysis function and return outputs
    corr_r_responseAccuracy, corr_p_responseAccuracy, fig_responseAccuracy, ax_responseAccuracy = relationships_sportRT(
        nc_data, 'responseAccuracy', 'Accuracy (Arcsine %)')

    # Save figure
    fig_responseAccuracy.savefig(os.path.join('..','results', 'figures', 'scatter_response-accuracy_SportRT.png'),
                                 format='png', dpi=600)

    # Close figure
    plt.close(fig_responseAccuracy)

# %% ---------- end of neurocog_relationships.py ---------- %% #
