from plots.plot_dynamics import *
from simulations.data_synchro_transition_winfree import *
from simulations.data_synchro_transition_kuramoto\
    import *
import matplotlib
import tkinter.simpledialog
from tkinter import messagebox
import time


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
first_community_color = "#2171b5"  # Old color #064878 # 6, 72, 120
second_community_color = "#f16913"  # Old color "#ff370f"  # 255, 55, 15
markersize = 3
fontsize = 18
fontsize_legend = 18
labelsize = 18
linewidth = 2


def plot_complete_vs_reduced_vs_pout(p_out_array, complete_R_array,
                                     reduced_R_array, complete_color="#2171b5",
                                     reduced_color="#9ecae1", fontsize = 18,
                                     fontsize_legend = 18,labelsize = 18,
                                     linewidth=2):
    return


p_array = np.linspace(0.001, 1, 50)  # p_out


fig = plt.figure(figsize=(12, 4))
# width = 7.057
# height = width/3.2
# fig = plt.figure(figsize=(width, height))
# axes = fig.subplots(1,3, sharey=True)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

ax1 = plt.subplot(3, 4, 1)

# Winfree on SBM
plt.fill_between(p_array, mean_r_wb2 - std_r_wb2, mean_r_wb2 + std_r_wb2,
                 color="k", alpha=0.1)
plt.plot(p_array, mean_r_wb2, color="k",
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R_wb2, yerr=std_R_wb2, color="#9b9b9b",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
ylab = plt.ylabel("$R$", fontsize=fontsize)
ylab.set_rotation(0)
ax1.yaxis.set_label_coords(-0.2, 0.3)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax1.set_xticklabels([])

ax5 = plt.subplot(3, 4, 5)

plt.fill_between(p_array, mean_r1_wb2 - std_r1_wb2, mean_r1_wb2 + std_r1_wb2,
                 color=first_community_color, alpha=0.1)
plt.plot(p_array, mean_r1_wb2, color=first_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R1_wb2, yerr=std_R1_wb2, color="lightblue",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
ylab2 = plt.ylabel("$R_1$", fontsize=fontsize)
ylab2.set_rotation(0)
ax5.yaxis.set_label_coords(-0.2, 0.3)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax5.set_xticklabels([])

ax9 = plt.subplot(3, 4, 9)
plt.fill_between(p_array, mean_r2_wb2 - std_r2_wb2, mean_r2_wb2 + std_r2_wb2,
                 color=second_community_color, alpha=0.1)
plt.plot(p_array, mean_r2_wb2, color=second_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R2_wb2, yerr=std_R2_wb2, color="#feb24c",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
ylab3 = plt.ylabel("$R_2$", fontsize=fontsize)
ylab3.set_rotation(0)
ax9.yaxis.set_label_coords(-0.2, 0.3)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()


ax2 = plt.subplot(3, 4, 2)

p_out_array = np.linspace(0.01, 1, 50)

# Kuramoto on SBM
plt.fill_between(p_out_array, mean_r_kb - std_r_kb, mean_r_kb + std_r_kb,
                 color="k", alpha=0.1)
plt.plot(p_out_array, mean_r_kb, color="k",
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_out_array, mean_R_kb, yerr=std_R_kb, color="#9b9b9b",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax2.set_xticklabels([])
ax2.set_yticklabels([])

ax6 = plt.subplot(3, 4, 6)
plt.fill_between(p_out_array, mean_r1_kb - std_r1_kb, mean_r1_kb + std_r1_kb,
                 color=first_community_color, alpha=0.1)
plt.plot(p_out_array, mean_r1_kb, color=first_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_out_array, mean_R1_kb, yerr=std_R1_kb, color="lightblue",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax6.set_xticklabels([])
ax6.set_yticklabels([])

ax10 = plt.subplot(3, 4, 10)
plt.fill_between(p_out_array, mean_r2_kb - std_r2_kb, mean_r2_kb + std_r2_kb,
                 color=second_community_color, alpha=0.1)
plt.plot(p_out_array, mean_r2_kb, color=second_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_out_array, mean_R2_kb, yerr=std_R2_kb, color="#feb24c",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
plt.xlabel("$p_{out}$", fontsize=fontsize)
ax10.xaxis.set_label_coords(1.05, -0.5)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax10.set_yticklabels([])


ax3 = plt.subplot(3, 4, 3)

# Winfree on SBM
plt.fill_between(p_array, mean_r_ws - std_r_ws, mean_r_ws + std_r_ws,
                 color="k", alpha=0.1)
plt.plot(p_array, mean_r_ws, color="k",
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R_ws, yerr=std_R_ws, color="#9b9b9b",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax3.set_xticklabels([])
ax3.set_yticklabels([])


ax7 = plt.subplot(3, 4, 7)
plt.fill_between(p_array, mean_r1_ws - std_r1_ws, mean_r1_ws + std_r1_ws,
                 color=first_community_color, alpha=0.1)
plt.plot(p_array, mean_r1_ws, color=first_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R1_ws, yerr=std_R1_ws, color="lightblue",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax7.set_xticklabels([])
ax7.set_yticklabels([])


ax11 = plt.subplot(3, 4, 11)
plt.fill_between(p_array, mean_r2_ws - std_r2_ws, mean_r2_ws + std_r2_ws,
                 color=second_community_color, alpha=0.1)
plt.plot(p_array, mean_r2_ws, color=second_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R2_ws, yerr=std_R2_ws, color="#feb24c",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
#plt.xlabel("$p_{out}$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([0, 1.02])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax11.set_yticklabels([])

ax4 = plt.subplot(3, 4, 4)

# Kuramoto on SBM
plt.fill_between(p_array, mean_r_ks - std_r_ks, mean_r_ks + std_r_ks,
                 color="k", alpha=0.1)
plt.plot(p_array, mean_r_ks, color="k",
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R_ks, yerr=std_R_ks, color="#9b9b9b",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax4.set_xticklabels([])
ax4.set_yticklabels([])

ax8 = plt.subplot(3, 4, 8)
plt.fill_between(p_array, mean_r1_ks - std_r1_ks, mean_r1_ks + std_r1_ks,
                 color=first_community_color, alpha=0.1)
plt.plot(p_array, mean_r1_ks, color=first_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R1_ks, yerr=std_R1_ks, color="lightblue",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax8.set_xticklabels([])
ax8.set_yticklabels([])

ax12 = plt.subplot(3, 4, 12)
plt.fill_between(p_array, mean_r2_ks - std_r2_ks, mean_r2_ks + std_r2_ks,
                 color=second_community_color, alpha=0.1)
plt.plot(p_array, mean_r2_ks, color=second_community_color,
         linewidth=linewidth, label=r"Complete")
plt.errorbar(p_array, mean_R2_ks, yerr=std_R2_ks, color="#feb24c",
             linewidth=linewidth - 1, linestyle='--', marker="o",
             markersize=markersize, dash_capstyle='butt',
             dash_joinstyle="bevel", label=r"Reduced")
# plt.xlabel("$p_{out}$", fontsize=fontsize)
plt.tick_params(axis='both', which='major', labelsize=labelsize)
plt.ylim([-0.1, 1.1])
plt.xlim([0, 1.02])
# plt.legend(loc=4, fontsize=fontsize_legend)
plt.tight_layout()
ax12.set_yticklabels([])

plt.subplots_adjust(left=0.07, bottom=0.17, right=0.98,
                    top=0.9, wspace=0.15, hspace=0.25)

ax1.title.set_text('(a) Winfree')
ax2.title.set_text('(b) Kuramoto')
ax3.title.set_text('(c) Winfree')
ax4.title.set_text('(d) Kuramoto')
ax1.title.set_fontsize(fontsize)
ax2.title.set_fontsize(fontsize)
ax3.title.set_fontsize(fontsize)
ax4.title.set_fontsize(fontsize)

plt.show()
if messagebox.askyesno("Python",
                       "Would you like to save the plot ?"):
    window = tkinter.Tk()
    window.withdraw()  # hides the window
    file = tkinter.simpledialog.askstring("File: ", "Enter your file name")

    timestr = time.strftime("%Y_%m_%d_%Hh%Mmin%Ssec")

    fig.savefig("data/article_figures/"
                "{}_{}_complete_vs_reduced_winfree_kuramoto"
                ".png".format(timestr, file))
    fig.savefig("data/article_figures/"
                "{}_{}_complete_vs_reduced_winfree_kuramoto"
                ".pdf".format(timestr, file))


# Netsci 2019

# fig = plt.figure(figsize=(10, 8))
#
# # Theta model
# p_array = np.linspace(0.001, 1, 50)
# with open('data/theta/2019_04_09_13h06min23sec_theta_result'
#           '_complete_r_list.json') as json_data:
#     r_list = np.array(json.load(json_data))
# with open('data/theta/2019_04_09_13h06min23sec_theta_result'
#           '_reduced_R_list.json') as json_data:
#     R_list = np.array(json.load(json_data))
#
# plt.subplot(311)
# plt.plot(p_array, r_list, color=first_community_color,
#          linestyle='-', linewidth=linewidth, label="$Complete$")
# plt.plot(p_array, R_list, color="lightblue",
#          linestyle='--', linewidth=linewidth, label='$Reduced$')
# #plt.scatter(p_array, r_list, s=100, color=first_community_color,
# #            label="$Complete$")
# #plt.scatter(p_array, R_list, s=50, color="lightblue",
# #            label="$Reduced$", marker='s')
# plt.ylabel("$|Z|$", fontsize=fontsize)
# plt.xlabel("$p$", fontsize=fontsize)
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.ylim([0, 1.02])
# plt.xlim([0, 1.02])
# #plt.legend(loc=3, fontsize=fontsize_legend)
# # , ncol=2)# bbox_to_anchor=(1.35, 1.01),
#
#
#
# # Cowan-Wilson (Firing-rate)
# pout_array = np.linspace(0.01, 1, 500)
# with open('data/cowan_wilson/2019_05_15_21h43min25sec_very_long_simulation_'
#           'complete_r1_list.json') as json_data:
#     R1c_list = np.array(json.load(json_data))
# with open('data/cowan_wilson/2019_05_15_21h43min25sec_very_long_simulation_'
#           'complete_r2_list.json') as json_data:
#     R2c_list = np.array(json.load(json_data))
# with open('data/cowan_wilson/2019_05_15_21h43min25sec_very_long_simulation_'
#           'reduced_R1_list.json') as json_data:
#     R1r_list = np.array(json.load(json_data))
# with open('data/cowan_wilson/2019_05_15_21h43min25sec_very_long_simulation_'
#           'reduced_R2_list.json') as json_data:
#     R2r_list = np.array(json.load(json_data))
#
#
# plt.subplot(312)
# plt.plot(pout_array, R1c_list, linestyle='-', color=first_community_color,
#          label="$Complete$", linewidth=linewidth)
# plt.plot(pout_array, R1r_list, linestyle='--', color="lightblue",
#          label="$Reduced$", linewidth=linewidth)
# plt.plot(pout_array, R2c_list, linestyle='-', color=second_community_color,
#          label="$Complete$", linewidth=linewidth)
# plt.plot(pout_array, R2r_list, linestyle='--', color="#ffa159",
#          label="$Reduced$", linewidth=linewidth)
# plt.ylabel("$R_{\\mu}$", fontsize=fontsize)
# plt.xlabel("$p_{out}$", fontsize=fontsize)
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.ylim([0, 1.02])
# plt.xlim([0, 1.02])
# #plt.legend(loc=3, fontsize=fontsize_legend)
# # , ncol=2)# bbox_to_anchor=(1.35, 1.01),
#
#
# # Kuramoto
# w_array = np.linspace(0, 10, 50)
# with open('data/kuramoto/2019_05_23_17h18min16sec_very_nice_scc_transition_'
#           'chimera_to_partial_synchrony_mean_Rp1_complete_vs_s.json') \
#         as json_data:
#     Rp1c_list = np.array(json.load(json_data))
# with open('data/kuramoto/2019_05_23_17h18min16sec_very_nice_scc_transition_'
#           'chimera_to_partial_synchrony_mean_Rp1_reduced_vs_s.json') \
#         as json_data:
#     Rp1r_list = np.array(json.load(json_data))
# with open('data/kuramoto/2019_05_23_17h18min16sec_very_nice_scc_transition_'
#           'chimera_to_partial_synchrony_mean_Rp2_complete_vs_s.json') \
#         as json_data:
#     Rp2c_list = np.array(json.load(json_data))
# with open('data/kuramoto/2019_05_23_17h18min16sec_very_nice_scc_transition_'
#           'chimera_to_partial_synchrony_mean_Rp2_reduced_vs_s.json') \
#         as json_data:
#     Rp2r_list = np.array(json.load(json_data))
#
# plt.subplot(313)
# plt.plot(w_array, Rp1c_list, linestyle='-', color=first_community_color,
#          label="$Complete$", linewidth=linewidth)
# plt.plot(w_array, Rp1r_list, linestyle='--', color="lightblue",
#          label="$Reduced$", linewidth=linewidth)
# plt.plot(w_array, Rp2c_list, linestyle='-', color=second_community_color,
#             label="$Complete$", linewidth=linewidth)
# plt.plot(w_array, Rp2r_list, linestyle='--', color="#ffa159",
#             label="$Reduced$", linewidth=linewidth)
# plt.ylabel("$|Z_{\\mu}|$", fontsize=fontsize)
# plt.xlabel("$w$", fontsize=fontsize)
# plt.tick_params(axis='both', which='major', labelsize=labelsize)
# plt.ylim([0, 1.02])
# plt.xlim([0, 10.02])
# #plt.legend(loc=3, fontsize=fontsize_legend)
# # , ncol=2)# bbox_to_anchor=(1.35, 1.01),
#
#
# plt.show()
