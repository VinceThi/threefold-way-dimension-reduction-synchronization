import matplotlib.pyplot as plt
# Configuration d'un environnement LaTeX pour les graphiques générés avec matplotlib.pyplot
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams['font.style'] =  "normal"
plt.rcParams['font.variant'] = "normal"
plt.rcParams['font.weight'] = "medium"
plt.rcParams['font.stretch'] = "normal"
plt.rcParams['font.size'] = 15


# plt.rcParams['font.serif'] = DejaVu Serif, Bitstream Vera Serif, New Century Schoolbook, Century Schoolbook L, Utopia, ITC Bookman, Bookman, Nimbus Roman No9 L, Times New Roman, Times, Palatino, Charter, serif
# plt.rcParams['font.sans-serif'] = DejaVu Sans, Bitstream Vera Sans, Lucida Grande, Verdana, Geneva, Lucid, Arial, Helvetica, Avant Garde, sans-serif
# plt.rcParams['font.cursive'] = Apple Chancery, Textile, Zapf Chancery, Sand, Script MT, Felipa, cursive
# plt.rcParams['font.fantasy'] = Comic Sans MS, Chicago, Charcoal, Impact, Western, Humor Sans, xkcd, fantasy
# plt.rcParams['font.monospace'] = DejaVu Sans Mono, Bitstream Vera Sans Mono, Andale Mono, Nimbus Mono L, Courier New, Courier, Fixed, Terminal, monospace

font = {'family':'serif', 'serif': ['Times New Roman']}
plt.rc('font', **font)
plt.rc('text', usetex=True)
plt.rcParams['text.usetex'] = True


plt.rcParams['axes.facecolor'] = "white"        # axes background color
plt.rcParams['axes.edgecolor'] = "black"        # axes edge color
plt.rcParams['axes.linewidth'] = 0.8            # edge linewidth
plt.rcParams['axes.grid'] = False               # display grid or not
plt.rcParams['axes.titlesize'] = "large"        # fontsize of the axes title
plt.rcParams['axes.titlepad'] = 6               # pad between axes and title in points
plt.rcParams['axes.labelsize'] = 15             # fontsize of the x any y labels
plt.rcParams['axes.labelpad'] = 4            # space between label and axis
plt.rcParams['axes.labelweight'] = "normal"     # weight of the x and y labels
plt.rcParams['axes.labelcolor'] = "black"       # label color
plt.rcParams['axes.axisbelow'] = 'line'         # draw axis gridlines and ticks below

## XTICKS
plt.rcParams['xtick.top'] = True               # draw ticks on the top side
plt.rcParams['xtick.bottom'] = True             # draw ticks on the bottom side
plt.rcParams['xtick.major.size'] =  3.5         # major tick size in points
plt.rcParams['xtick.minor.size'] =  2           # minor tick size in points
plt.rcParams['xtick.major.width'] =  0.8        # major tick width in points
plt.rcParams['xtick.minor.width'] =  0.6        # minor tick width in points
plt.rcParams['xtick.major.pad'] =  3.5          # distance to major tick label in points
plt.rcParams['xtick.minor.pad'] =  3.4          # distance to the minor tick label in points
plt.rcParams['xtick.color'] =  "k"              # color of the tick labels
plt.rcParams['xtick.labelsize'] =  15           # fontsize of the tick labels
plt.rcParams['xtick.direction'] =  'in'        # direction: in, out, or inout
plt.rcParams['xtick.major.top'] = True          # draw x axis top major ticks
plt.rcParams['xtick.major.bottom'] = True       # draw x axis bottom major ticks
plt.rcParams['xtick.minor.top'] = True          # draw x axis top minor ticks
plt.rcParams['xtick.minor.bottom'] = True       # draw x axis bottom minor ticks

## YTICKS
plt.rcParams['ytick.left'] = True               # draw ticks on the left side
plt.rcParams['ytick.right'] = True            # draw ticks on the right side
plt.rcParams['ytick.major.size'] = 3.5        # major tick size in points
plt.rcParams['ytick.minor.size'] =  2           # minor tick size in points
plt.rcParams['ytick.major.width'] = 0.8         # major tick width in points
plt.rcParams['ytick.minor.width'] =  0.6        # minor tick width in points
plt.rcParams['ytick.major.pad'] =  3.5          # distance to major tick label in points
plt.rcParams['ytick.minor.pad'] =  3.4          # distance to the minor tick label in points
plt.rcParams['ytick.color'] = "k"               # color of the tick labels
plt.rcParams['ytick.labelsize'] =  15           # fontsize of the tick labels
plt.rcParams['ytick.direction'] =  "in"        # direction: in, out, or inout
plt.rcParams['ytick.minor.visible'] = False     # visibility of minor ticks on y-axis
plt.rcParams['ytick.major.left'] =  True        # draw y axis left major ticks
plt.rcParams['ytick.major.right'] = True        # draw y axis right major ticks
plt.rcParams['ytick.minor.left'] =  True        # draw y axis left minor ticks
plt.rcParams['ytick.minor.right'] =  True       # draw y axis right minor ticks

## GRIDS
plt.rcParams['grid.linestyle'] = "-"        # solid
plt.rcParams['grid.linewidth'] = 0.8       # in points
plt.rcParams['grid.alpha'] = 1.0       # transparency, between 0.0 and 1.0

## Legend
plt.rcParams['legend.loc'] =  "upper right"
plt.rcParams['legend.frameon'] = True     # if True, draw the legend on a background patch
plt.rcParams['legend.framealpha'] = 1      # legend patch transparency
plt.rcParams['legend.facecolor'] = "inherit"  # inherit from axes.facecolor; or color spec
plt.rcParams['legend.edgecolor'] = "black"      # background patch boundary color
plt.rcParams['legend.fancybox'] = False     # if True, use a rounded box for the
plt.rcParams['legend.shadow'] = False    # if True, give background a shadow effect
plt.rcParams['legend.numpoints'] = 1        # the number of marker points in the legend line
plt.rcParams['legend.scatterpoints'] = 1        # number of scatter points
plt.rcParams['legend.markerscale'] = 1.0      # the relative size of legend markers vs. original
plt.rcParams['legend.fontsize'] =  15
plt.rcParams['legend.borderpad'] = 0.4      # border whitespace
plt.rcParams['legend.labelspacing'] = 0.5      # the vertical space between the legend entries
plt.rcParams['legend.handlelength'] = 2.0      # the length of the legend lines
plt.rcParams['legend.handleheight'] = 0.7      # the height of the legend handle
plt.rcParams['legend.handletextpad'] = 0.8      # the space between the legend line and legend text
plt.rcParams['legend.borderaxespad'] = 0.5      # the border between the axes and legend edge
plt.rcParams['legend.columnspacing'] = 2.0      # column separation

## FIGURE
plt.rcParams['figure.titlesize'] = "large"     # size of the figure title (Figure.suptitle())
plt.rcParams['figure.titleweight'] = "normal"   # weight of the figure title
plt.rcParams['figure.figsize'] = [8,6]   # figure size in inches
plt.rcParams['figure.dpi'] = 100      # figure dots per inch
plt.rcParams['figure.facecolor'] = "white"   # figure facecolor; 0.75 is scalar gray
plt.rcParams['figure.edgecolor'] = "white"   # figure edgecolor
plt.rcParams['figure.autolayout'] = True  # When True, automatically adjust subplot
