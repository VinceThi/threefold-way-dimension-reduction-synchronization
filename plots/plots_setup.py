# -*- coding: utf-8 -*-
# @author: Vincent Thibeault

import matplotlib.colors
import matplotlib.pyplot as plt

first_community_color = "#2171b5"           # RGB: 33, 113, 181    blue
second_community_color = "#f16913"          # RGB: 241, 105, 19    orange
third_community_color = "#238b45"           # RGB: 35, 139, 69     green
fourth_community_color = "#6a51a3"          # RGB: 106, 81, 163    purple
reduced_first_community_color = "#9ecae1"   # RGB: 158, 202, 225   light blue
reduced_second_community_color = "#fdd0a2"  # RGB: 253, 208, 162   light orange
reduced_third_community_color = "#a1d99b"   # RGB: 161, 217, 155   light green
reduced_fourth_community_color = "#9e9ac8"  # RGB: 158, 154, 200   light purple
complete_grey = "#252525"                   # RGB: 37, 37, 37      light grey
reduced_grey = "#969696"                    # RGB: 150, 150, 150   grey
total_color = "#525252"                     # RGB: 82, 82, 82      grey
fontsize = 12
inset_fontsize = 9
fontsize_legend = 12
labelsize = 12
inset_labelsize = 9
linewidth = 2
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.unicode'] = True
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
# # plot parameters
# font_size=8
# plt.style.use('seaborn-paper')
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', serif='Computer Modern')
# plt.rc('xtick', labelsize=font_size)
# plt.rc('ytick', labelsize=font_size)
# plt.rc('axes', labelsize=font_size)


# IMPORTANT : sur les colormaps: Ne pas utiliser jet !
#             Batelot(?), viridis, turbo sont biens
# (mais attention, turbo n'est pas conceptuellement uniforme)
# Voir https://jiffyclub.github.io/palettable/
# https://clauswilke.com/dataviz/aesthetic-mapping.html

cdict = {
    'red':   ((0, 255/255, 255/255),
              (0.4, 253 / 255, 253 / 255),
              (0.6, 253 / 255, 253 / 255),
              (0.8, 253 / 255, 253 / 255),
              (1.0, 241 / 255, 241 / 255),),
    'green': ((0, 245/255, 245/255),
              (0.4, 208 / 255, 208 / 255),
              (0.6, 174 / 255, 174 / 255),
              (0.8, 141 / 255, 141 / 255),
              (1.0, 105 / 255, 105 / 255),),
    'blue':  ((0, 235/255, 235/255),
              (0.4, 162 / 255, 162 / 255),
              (0.6, 107 / 255, 107 / 255),
              (0.8, 60 / 255,  60 / 255),
              (1.0, 19 / 255, 19 / 255))
}
cm = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict, 1024)

cdict2 = {
    'red':   ((0,   241 / 255, 241 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0,  33 / 255,  33 / 255),),
    'green': ((0,   105 / 255, 105 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 113 / 255, 113 / 255),),
    'blue':  ((0,    19 / 255,  19 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 181 / 255, 181 / 255))
}
cm2 = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict2, 1024)

cdict3 = {
    'red':   ((0,   158 / 255, 158 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 253 / 255, 253 / 255),),
    'green': ((0,   202 / 255, 202 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 208 / 255, 208 / 255),),
    'blue':  ((0,   225 / 255, 225 / 255),
              (0.5, 255 / 255, 255 / 255),
              (1.0, 162 / 255, 162 / 255))
}
cm3 = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict3, 1024)


cdict4 = {
    'red':   ((0,   255 / 255, 255 / 255),
              (0.1, 158 / 255, 158 / 255),
              (1.0, 253 / 255, 253 / 255),),
    'green': ((0,   255 / 255, 255 / 255 ),
              (0.1, 202 / 255, 202 / 255),
              (1.0, 208 / 255, 208 / 255),),
    'blue':  ((0,   255 / 255, 255 / 255),
              (0.1, 225 / 255, 225 / 255),
              (1.0, 162 / 255, 162 / 255))
}
cm4 = matplotlib.colors.LinearSegmentedColormap('my_colormap', cdict4, 1024)
