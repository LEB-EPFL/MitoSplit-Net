import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=18)
plt.rc('ytick', labelsize=18)
plt.rc('axes', labelsize=20)
plt.rc('legend', fontsize=18)

def plot_merge(input_img, output_img, cmap=['gray', 'inferno'], title='Mito + GT', 
               alpha=0.5, frame=None, ax=None):
  title_size = 20
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
  ax.axis('off')
  return ax
  
def plot_comparison(input_img, output_img, cmap = ['gray', 'inferno'],
                    labels=['input_data', 'output_data'], frame=None, 
                    merge=False, axes=None):
  title_size = 20
  
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

def plot_histogram(bincenters, hist, ax=None, width=None, xlabel=r'$t$', ylabel=r'$f$', edgecolor='white', **kwargs):
  if ax is None:
    ax = plt.subplots(figsize=(6, 6))[1]
  if width is None:
    width = bincenters[1]-bincenters[0]
    
  barplot = ax.bar(bincenters, hist, width=width, edgecolor=edgecolor, ecolor='black', capsize=3, **kwargs)
  ax.set(xlabel=xlabel, ylabel=ylabel)
  return ax
