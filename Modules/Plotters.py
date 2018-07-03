import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style("whitegrid")

def getHistoryPlot(histories):
    plt.figure(figsize=(16,8))

    for i, history in enumerate(histories):
        if i == 0:
            plt.plot(history['loss'], color='g', label='Training')
            plt.plot(history['val_loss'], color='b', label='Validation')
        else:
            plt.plot(history['loss'], color='g')
            plt.plot(history['val_loss'], color='b')

    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Epoch", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()

def getModelHistoryComparisonPlot(histories, names, cv=False):
    plt.figure(figsize=(16,8))
    
    for i, (history, name) in enumerate(zip(histories, names)):
        if cv:
            sns.tsplot([history[x]['val_loss'] for x in range(len(history))], condition=name, color=sns.color_palette()[i])
        else:
            plt.plot(history['val_loss'], label=name)

    plt.legend(loc='best', fontsize=16)
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Epoch", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()

def getLRFinderComparisonPlot(lrFinders, names, logX=True, logY=True):
    plt.figure(figsize=(16,8))
    
    for lrFinder, name in zip(lrFinders, names):
        plt.plot(lrFinder.history['lr'], lrFinder.history['val_loss'], label=name)

    plt.legend(loc='best', fontsize=16)
    if logX: plt.xscale('log')
    if logY: plt.yscale('log')
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    plt.xlabel("Learning rate", fontsize=24, color='black')
    plt.ylabel("Loss", fontsize=24, color='black')
    plt.show()

def getMonitorComparisonPlot(monitors, names, xAxis='iter', yAxis='Loss', lrLogX=True, logY=True):
    plt.figure(figsize=(16,8))
    for monitor, name in zip(monitors, names):
        if isinstance(monitor.history['val_loss'][0], list):
            if yAxis == 'Loss':
                y = np.array(monitor.history['val_loss'])[:,0]
            else:
                y = np.array(monitor.history['val_loss'])[:,1]
        else:
            y = monitor.history['val_loss']
                
        if xAxis == 'iter':
            plt.plot(range(len(monitor.history['val_loss'])), y, label=name)
        elif xAxis == 'mom':
            plt.plot(monitor.history['mom'], y, label=name)
        else:
            plt.plot(monitor.history['lr'], y, label=name)

    plt.legend(loc='best', fontsize=16)
    if lrLogX: plt.xscale('log')
    if logY: plt.yscale('log')
    plt.xticks(fontsize=16, color='black')
    plt.yticks(fontsize=16, color='black')
    if xAxis == 'iter':
        plt.xlabel("Iteration", fontsize=24, color='black')
    elif xAxis == 'mom':
        plt.xlabel("Momentum", fontsize=24, color='black')
    else:
        plt.xlabel("Learning rate", fontsize=24, color='black')
    plt.ylabel(yAxis, fontsize=24, color='black')
    plt.show()