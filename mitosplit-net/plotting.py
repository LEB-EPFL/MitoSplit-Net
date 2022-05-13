import numpy as np
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=18)

def plot_merge(input_img, output_img, cmap=None, title='Mito + GT', 
               alpha=0.5, frame=None, ax=None):
  title_size = 20
  if cmap is None:
    cmap = ['gray', 'inferno']
  if type(cmap)==str:
    cmap = [cmap]*2
  if ax is None:
    fig, ax = plt.subplots(figsize=(5,5))
  else:
    fig = ax.figure
  if frame is None:
    ax.set_title(title, size=title_size)
  else:
    ax.set_title(title+' [%i]'%frame, size=title_size)
  ax.imshow(input_img, cmap=cmap[0])
  ax.imshow(output_img, cmap=cmap[1], alpha=alpha)
  ax.set(xticks=[], yticks=[])
  return ax
  
def plot_comparison(input_img, output_img, cmap = ['gray', 'inferno'],
                    labels=['input_data', 'output_data'], frame=None, 
                    merge=False, axes=None):
  title_size = 20
  if type(cmap)==str:
    cmap = [cmap]*2
  if merge:
    if axes is None:
      fig, axes = plt.subplots(1, 3, figsize=(5*3, 5))
    _ = plot_merge(input_img, output_img, cmap=cmap, title='Merge', frame=frame, ax=axes[2])
  
  else:
    if axes is None:
      fig, axes = plt.subplots(1, 2, figsize=(5*2, 5))
  
  for ax, img, c, title in zip(axes[:2], [input_img, output_img], 
                               cmap, labels[:2]):
    if frame is None:
      ax.set_title(title, size=title_size)
    else:
      ax.set_title(title+' [%i]'%frame, size=title_size)
    
    ax.imshow(img, cmap=c)
    ax.axis('off')
  plt.subplots_adjust(wspace=0.05)
  return axes

def plot_outputs(input_test, output_test, pred_output_test, frames_test, nb_examples=3, cmap=None, title=None):
    nb_img = output_test.shape[0]
    item_id = np.sort(np.random.randint(0, nb_img, nb_examples))
    fig, axes = plt.subplots(nb_examples, 2, figsize=(5*2, 5*nb_examples))
    fig.text(0.5, 0.92, title, fontsize=20, ha='center')
    for i, ax in zip(item_id, axes):
        plot_merge(input_test[i], output_test[i], title='output_test', 
                   cmap=cmap, frame=frames_test[i], ax=ax[0])
        plot_merge(input_test[i], pred_output_test[i], title='pred_output_test', 
                   cmap=cmap, frame=frames_test[i], ax=ax[1])
    fig.subplots_adjust(wspace=0, hspace=0.15, bottom=0.2, top=0.9)
    return axes

def plot_histogram(bincenters, hist, ax=None, width=None, xlabel=r'$t$', ylabel=r'$f$', edgecolor='white', **kwargs):
  if ax is None:
    ax = plt.subplots(figsize=(6, 6))[1]
  if width is None:
    width = bincenters[1]-bincenters[0]
    
  barplot = ax.bar(bincenters, hist, width=width, edgecolor=edgecolor, ecolor='black', capsize=3, **kwargs)
  ax.set(xlabel=xlabel, ylabel=ylabel)
  return ax

def plot_metrics(metrics, color=None, title=None, ylim=None, loc=(0.56, 0.75), ncol=None, legend=True, ax=None, **kwargs):
  fontsize = 20
  N = len(metrics)
  if ax is None:
    fig, ax = plt.subplots(figsize=(N, 5))
  else:
    fig = ax.figure
  if color is None:
    color = plt.cm.get_cmap('tab10')(range(N))  
  if ylim is None:
    ylim = [0, 1.1*metrics.max()]
  
  xticks = range(N)
  yticks = np.linspace(ylim[0], ylim[1], 4, endpoint=True)
  vals = metrics.values
  ax.set_title(title, size=18)
  ax.bar(xticks, vals, color=color, **kwargs)
  ax.set(xticks=xticks, yticks=yticks)
  ax.set(xticklabels=[], yticklabels=yticks)
  ax.set_ylim(ylim)  
  ax.set(xticks=[])
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)
  if legend:
    if ncol is not None:
      ax.legend(handles=[plt.bar([0], [0], color=c) for c in color], labels=list(metrics.index), 
                loc=loc, framealpha=1, fontsize=fontsize, ncol=ncol)
    else:
      ax.legend(handles=[plt.bar([0], [0], color=c) for c in color], labels=list(metrics.index), 
          loc=loc, framealpha=1, fontsize=fontsize)
  ax.yaxis.set_major_formatter('{x:1.2f}')
  return ax

def plot_metrics_comparison(metrics, color=None, title=None, ylim=None, ax=None, xscale = 1.5, yscale=1, legend=True, **kwargs):
  N, n = metrics.shape
  if ax is None:
    fig, ax = plt.subplots(figsize=(N*xscale, 6*yscale))
  else:
    fig = ax.figure
  if color is None:
    color = plt.cm.get_cmap('tab10')(np.linspace(0, n*0.1, n))  
  if ylim is None:
    ylim = [0, 1.1*metrics.max().max()]
    
  keys = list(metrics.index)
  i = np.arange(N)
  width = 0.7*(i[1]-i[0])/n
  for j, metric_name in enumerate(metrics):
    vals = metrics[metric_name]
    ax.bar(i + width*(j-(n-1)/2), vals, width=width, align='center', 
            color=color[j], label=metric_name, **kwargs)

  ax.set_title(title, size=24)
  ax.set_xticks(i)
  ax.set_xticklabels(keys)
  ax.set_ylim(ylim)  
  fig.subplots_adjust(top=0.78)
  if legend:
    fig.legend(bbox_to_anchor=(1, 0.73), loc=2, borderaxespad=-1)#, ncol=n//2 + 1)
    
  return ax


def plot_performance_curves(metrics, output_test, colors=None, axes=None):
    
    precRecRand = {}
    for model_name in metrics:
        out_binary = output_test[model_name]>0
        precRecRand[model_name] = out_binary.sum()/(out_binary.size)

    print('\nFraction of positives')
    print(precRecRand)
        
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True, sharey=True)
    else:
        fig = axes[0].figure

    if colors is None:
        colors = plt.cm.get_cmap('tab10')(range(len(metrics)))

    for model_name, color in zip(metrics, colors):
        axes[0].plot(metrics[model_name]['FPR'], metrics[model_name]['TPR'], label='s=%s'%model_name.split('_s')[-1])  
        axes[1].plot(metrics[model_name]['TPR'], metrics[model_name]['precision'], color=color)
        axes[1].axhline(precRecRand[model_name], ls='--', color=color)

    axes[0].plot(metrics[model_name]['FPR'], metrics[model_name]['FPR'], ls='--', color='black')
    axes[0].set_title('ROC', size=20)
    axes[0].set(xlabel='FPR', ylabel='TPR')

    axes[1].set_title('Precision-Recall', size=20)
    axes[1].set(xlabel='TPR', ylabel='Precision')

    for ax in axes:
      ax.set(xlim=[0, 1], ylim=[0, 1])
    fig.subplots_adjust(wspace=0.25)
    fig.legend(loc='upper right', bbox_to_anchor=(1.3, 0.9), framealpha=0)
    plt.show()
    
    return axes
    