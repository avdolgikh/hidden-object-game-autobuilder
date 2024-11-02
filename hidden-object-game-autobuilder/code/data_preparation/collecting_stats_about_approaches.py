import os
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import ticker
from texttable import Texttable

METRIC_NAMES = \
    {
        "gradient5_color5_similarity": "Grad & color (1:1)",
        "gradient7_color3_similarity": "Grad & color (7:3)",
        "gradient_color_geom_similarity": "Grad, color & contour (4:3:3)",        
        "mimicry334_similarity": "Color, contour & MI (3:3:4)",
        "color30_similarity": "Color quantization",
        "ecc_grad_similarity": "ECC (grad)",
        "ssim_similarity": "SSIM",
        "mutual_information_similarity": "Mutual information",

        "abs_similarity": "L1 pixel-based",
        "ssd_similarity": "L2 pixel-based",
        "geometrical_similarity": "Contour-based",
        "gradient_similarity": "Gradient-based",

        "mimicry_similarity": "Color, contour & MI (4:3:3)",
        "gradient_color_similarity": "Grad & color (1:1)",        
        "color_similarity": "Color quantization",
        "wasserstein_similarity": "Wasserstein (EMD)",
        "ecc_gray_grad_similarity": "ECC (grayscale, grad)",
        "ecc_similarity": "ECC",
        "correlation_similarity": "Cross-correlation",

        "resnet_level4_similarity": "ResNet 4f", # end of res4
        "resnet_level3_similarity": "ResNet 4c_2b", # b/w "bn4c_branch2b" and "res4c_branch2c"
        "resnet_level2_similarity": "ResNet 3d_2a", # b/w bn3d_branch2a and res3d_branch2b

        "resnet_lowlevel_similarity": "ResNet 4f", # end of res4
        "resnet_similarity": "ResNet 5c", # "activation_49" (after "bn5c_branch2c") followed by GAP, end of res5

        "resnet_2_similarity": "ResNet 2c",
        "resnet_3_2_similarity": "ResNet 3b",
        "resnet_3_similarity": "ResNet 3d",
        "resnet_4_2_similarity": "ResNet 4b",
        "resnet_4_4_similarity": "ResNet 4d",
        "resnet_4_similarity": "ResNet 4f",
        "resnet_5_2_similarity": "ResNet 5b",
        "resnet_5_similarity": "ResNet 5c",
    }

METRIC_COLORS = {}



def filter_meta(folder, meta, is_negative_folder=False):
    ids = []
    for filename in os.listdir(folder):    
        id = os.path.splitext(filename)[0].split("_")[1]
        ids.append( id )
    if is_negative_folder:
        return meta.loc[ ~meta["id"].isin(ids) ]
    else:
        return meta.loc[ meta["id"].isin(ids) ]

def get_similarity_rates(similarities, bad, good):
    rates = {}

    for similarity in similarities:        
        total = 0    
        good_rate = 0    
        if similarity in bad:
            total += bad[similarity]
        if similarity in good:
            total += good[similarity]
            good_rate = good[similarity] / total

        rates[similarity] = np.round(good_rate, 2)

    return rates

def get_similarity_rates_by_field( field, bad, good ):
    bad_similarities = bad.groupby([field])["id"].count()
    good_similarities = good.groupby([field])["id"].count()
    similarities = list(set( bad_similarities.index.values.tolist() + good_similarities.index.values.tolist() ))
    return get_similarity_rates(similarities, bad_similarities, good_similarities)

def get_stats_for_global_similarity( global_similarity, rate, bad, good ):
    print("-----------------------------")
    print("Global similarity: {}. Rate: {}.".format( global_similarity, rate ) )

    bad = bad[ bad["gl_sim"] == global_similarity ]
    good = good[ good["gl_sim"] == global_similarity ]
    len_bad = len(bad)
    len_good = len(good)

    print("  Local similarity:")
    print("    Bad: yes - {}, no - {}.".format( np.round( bad["l_sim"].count() / len_bad, 2), np.round((len_bad - bad["l_sim"].count()) / len_bad, 2)))
    print("    Good: yes - {}, no - {}.".format( np.round( good["l_sim"].count() / len_good, 2), np.round((len_good - good["l_sim"].count()) / len_good, 2)))

    unique, counts = np.unique( bad["l_sim"].dropna().tolist(), return_counts=True )        
    print("    Local similarities in bad set: {}".format( dict(zip(unique, np.round(counts / counts.sum(), 2))) ) )
    unique, counts = np.unique( good["l_sim"].dropna().tolist(), return_counts=True )        
    print("    Local similarities in good set: {}".format( dict(zip(unique, np.round(counts / counts.sum(), 2))) ) )
    
    #def get_max_score(data):
    #    filter = data[ data["l_sim"].isna() ]
    #    return filter["score"].max() if len(filter) > 0 else data["score"].max()

    #print("  Score:")
    #print("    Max score in bad set: {}".format(bad[ bad["l_sim"].isna() ]["score"].max()) )
    #print("    Max score in good set: {}".format(good[ good["l_sim"].isna() ]["score"].max()) )
    
    print("  Sride:")
    unique, counts = np.unique( bad["stride"].tolist(), return_counts=True )        
    print("    Strides in bad set: {}".format( dict(zip(unique, np.round(counts / counts.sum(), 2))) ) )
    unique, counts = np.unique( good["stride"].tolist(), return_counts=True )
    print("    Strides in good set: {}".format( dict(zip(unique, np.round(counts / counts.sum(), 2))) ) )

    print("  Tau:")
    unique, counts = np.unique( bad["tau"].tolist(), return_counts=True )        
    print("    Tau in bad set: {}".format( dict(zip(unique, np.round(counts / counts.sum(), 2))) ) )
    unique, counts = np.unique( good["tau"].tolist(), return_counts=True )        
    print("    Tau in good set: {}".format( dict(zip(unique, np.round(counts / counts.sum(), 2))) ) )
    
    print("  Rotation:")
    good_rotation_angles = [ json.loads(coord)[2] for coord in good["coord"].tolist() ]
    good_rotation_angles_mean = np.round(np.mean(good_rotation_angles), 2)
    good_rotation_angles_std = np.round(np.std(good_rotation_angles), 2)
    print("    Rotation in good set: values - {}, mean - {}, std - {}.".format( good_rotation_angles, good_rotation_angles_mean, good_rotation_angles_std ) )
    bad_rotation_angles = [ json.loads(coord)[2] for coord in bad["coord"].tolist() ]
    bad_rotation_angles_mean = np.round(np.mean(bad_rotation_angles), 2)
    bad_rotation_angles_std = np.round(np.std(bad_rotation_angles), 2)
    print("    Rotation in bad set: values - {}, mean - {}, std - {}.".format( bad_rotation_angles, bad_rotation_angles_mean, bad_rotation_angles_std ) )
    
    print("  Scale factor:")
    good_scale_factors = [ json.loads(coord)[3] for coord in good["coord"].tolist() ]
    good_scale_factors_mean = np.round(np.mean(good_scale_factors), 2)
    good_scale_factors_std = np.round(np.std(good_scale_factors), 2)    
    print("    Scale factor in good set: values - {}, mean - {}, std - {}.".format( good_scale_factors, good_scale_factors_mean, good_scale_factors_std ) )
    bad_scale_factors = [ json.loads(coord)[3] for coord in bad["coord"].tolist() ]
    bad_scale_factors_mean = np.round(np.mean(bad_scale_factors), 2)
    bad_scale_factors_std = np.round(np.std(bad_scale_factors), 2)    
    print("    Scale factor in good bad: values - {}, mean - {}, std - {}.".format( bad_scale_factors, bad_scale_factors_mean, bad_scale_factors_std ) )

    results = { "good_rotation_angles": (good_rotation_angles_mean, good_rotation_angles_std),
                "bad_rotation_angles": (bad_rotation_angles_mean, bad_rotation_angles_std),
                "good_scale_factors": (good_scale_factors_mean, good_scale_factors_std),
                "bad_scale_factors": (bad_scale_factors_mean, bad_scale_factors_std),  }

    if "bg_hgt" in good.columns:
        print("  Resizing:")    
        good_bg_hgts = good["bg_hgt"].tolist()
        bad_bg_hgts = bad["bg_hgt"].tolist()
        good_bg_hgts_mean = np.round(np.mean(good_bg_hgts), 2)
        bad_bg_hgts_mean = np.round(np.mean(bad_bg_hgts), 2)
        print("    Resizing in good set, mean: {}.".format( good_bg_hgts_mean ) )
        print("    Resizing in bad set, mean: {}.".format( bad_bg_hgts_mean ) )
        results["good_bg_hgts_mean"] = good_bg_hgts_mean
        results["bad_bg_hgts_mean"] = bad_bg_hgts_mean

    return results

def calc_stats(bad, good, all, experiments):
    title = " search. Experiments: {}".format( experiments )

    global_rates = get_similarity_rates_by_field( "gl_sim", bad, good )
    plot_similarity_rates( global_rates, "Global" + title, "global_similarity_rates_{}.png".format( ",".join(map(str, experiments)) ) )
    
    stats = {}
    for global_similarity, global_rate in sorted(global_rates.items(), key=lambda x: x[1], reverse=True):        
        stats[global_similarity] = get_stats_for_global_similarity( global_similarity, global_rate, bad, good )
    plot_stats(stats, experiments)

    local_rates = get_similarity_rates_by_field( "l_sim", bad, good )
    plot_similarity_rates( local_rates, "Local" + title, "local_similarity_rates_{}.png".format( ",".join(map(str, experiments)) ) )
    time_stats = calc_quality_independent_stats(all, [local_similarity for local_similarity, rate in sorted(local_rates.items(), key=lambda x: x[1], reverse=True)])
    plot_time_stats(time_stats, "Time, min. Experiments: {}".format( experiments ), "time_{}.png".format( ",".join(map(str, experiments)) ))

    return global_rates

def calc_quality_independent_stats(meta, local_similarities):
    print("-----------------------------")
    print("Max scores:")
    time_stats = {}
    for local_similarity in local_similarities:
        scores = meta[ meta["l_sim"] == local_similarity ]["score"]
        max_score = scores.max()
        normalized_scores = scores.values / max_score        
        mean_time = int(np.round( meta[ meta["l_sim"] == local_similarity ]["time"].mean(), 0))
        time_stats[local_similarity] = mean_time
        print("  {}, max_score: {}, mean (normalized): {}, std (normalized): {}".format(    local_similarity,
                                                                                            max_score,
                                                                                            np.round(np.mean( normalized_scores ), 2),
                                                                                            np.round(np.std( normalized_scores ), 2)))
        print("      mean_time: {} s.".format( mean_time ))
    return time_stats

def get_stats(experiment):
    meta_file = '../../docs/exp{}_results_meta.csv'.format( experiment )
    good_folder = '../../data/Validation/exp_{}/Good'.format( experiment )
    bad_folder = '../../data/Validation/exp_{}/Not_Good'.format( experiment )

    meta = pd.read_csv(meta_file)
    good = filter_meta( good_folder, meta )
    bad = filter_meta( bad_folder, meta )
   
    calc_stats(bad, good)

def get_stats_batch(experiments, use_only_good_folder):
    good_set = []
    bad_set = []
    meta_set = []
    for exp_ind, experiment in enumerate(experiments):
        meta_file = '../../docs/exp{}_results_meta.csv'.format( experiment )
        good_folder = '../../data/Validation/exp_{}/Good'.format( experiment )
        bad_folder = '../../data/Validation/exp_{}/Not_Good'.format( experiment ) if not use_only_good_folder[exp_ind] else good_folder

        meta = pd.read_csv(meta_file)
        good_set.append( filter_meta( good_folder, meta ) )
        bad_set.append( filter_meta( bad_folder, meta, is_negative_folder=use_only_good_folder[exp_ind] ) )
        meta_set.append( meta )
    
    good = pd.concat( good_set )
    bad = pd.concat( bad_set )
    all = pd.concat( meta_set )

    global_rates = calc_stats(bad, good, all, experiments)
    return global_rates

def resnet_layers_stats(initial, tuned, improvement_factor):
    """
    Dependecy of rate (good or not good) from number of ResNet layer, which was used for calculating similarity
    """
    # https://realpython.com/iterate-through-dictionary-python/
    # https://webdevblog.ru/kogda-ispolzovat-list-comprehension-v-python/
    resnet_rates = { key: value for key, value in tuned.items() if "resnet" in key }
    resnet_rates["resnet_similarity"] = initial["resnet_similarity"] * improvement_factor
    return { METRIC_NAMES[key]: value for key, value in resnet_rates.items() }

def resnet_layers_stats_2(data):
    """
    Dependecy of rate (good or not good) from number of ResNet layer, which was used for calculating similarity
    """
    return { METRIC_NAMES[key] if key in METRIC_NAMES else key: value for key, value in data.items() if "resnet" in key }

def tuned_search_stats(initial, tuned):
    """
    Factors of rates increasing after move from stride=30 to stride=10 and from Boltzman dist to getting maximum.
    """    
    tuned = { METRIC_NAMES[key]: value for key, value in tuned.items() }
    initial = { METRIC_NAMES[key]: value for key, value in initial.items() }
    improvement_factors = { key: tuned[key] / initial[key] for key in (tuned.keys() & initial.keys()) }

    plot_improvement_factors( improvement_factors, "Factors of rates increasing with smaller strides, etc.", "improvement_factors.png" )    

    # to plot (??) functions set/lib: "print_improvement_factors"
    printable_improvement_factors = [['Metric', 'Improvement factor']] + [ [key, value] for key, value in improvement_factors.items() ]
    t = Texttable()
    t.add_rows(printable_improvement_factors)
    print(t.draw())

    return improvement_factors

def resnet_results(experiments, use_only_good_folder):
    resnet_exp = get_stats_batch( experiments, use_only_good_folder )
    plot_resnet_layers_stats(   resnet_layers_stats_2(resnet_exp),
                                "HOG rates of ResNet layers. Experiments: {}".format( experiments ),
                                "Layers",
                                "Rates",
                                "resnet_layers_{}.png".format( ",".join(map(str, experiments)) ))

# ==================== PLOT ============================
def get_colors(metrics):
    colors = []
    for similarity in metrics:
        if similarity not in METRIC_COLORS:
            METRIC_COLORS[similarity] = np.random.rand(3)
        colors.append( METRIC_COLORS[similarity] )
    return colors

def plot_similarity_rates( rates, title, file_name ):
    figure = plt.figure(figsize=(6, 4), dpi=200)
    ax = figure.add_subplot(1, 1, 1)
    ax.grid(which='major', alpha=0.3)

    rates = [ (global_similarity, rate) for global_similarity, rate in sorted(rates.items(), key=lambda x: x[1])] #, reverse=True
    metrics = [    METRIC_NAMES[ rate[0] ] if rate[0] in METRIC_NAMES else rate[0]
                            for rate in rates ]    
    rates = [ rate[1] for rate in rates ]
    ax.xaxis.set_major_formatter( ticker.PercentFormatter(1.0) )
    ax.barh( metrics, rates, color=get_colors(metrics))
    #ax.xticks(rotation=45)
    plt.title( title )
    #plt.show()
    
    with open('../../docs/stats_viz/{}'.format(file_name), 'wb') as file:
        figure.savefig(file, bbox_inches='tight')
        plt.close(figure)

def plot_similarity_double_rates( rates_set, title, file_name, legend=[] ):
    figure = plt.figure(figsize=(6, 4), dpi=200)
    ax = figure.add_subplot(1, 1, 1)
    ax.xaxis.set_major_formatter( ticker.PercentFormatter(1.0) )
    ax.grid(which='major', alpha=0.3)

    sorted_rates = [ (global_similarity, rate) for global_similarity, rate in sorted(rates_set[1].items(), key=lambda x: x[1])]
    ordered_similarities = [ rate[0] for rate in sorted_rates ]

    ind = np.arange(len(ordered_similarities))
    width = 0.3
    
    for i, rates in enumerate(rates_set):
        rates = [ rates[similarity] for similarity in ordered_similarities ]
        ax.barh(ind + width * ( i - 0.5), rates, width, label=legend)
        
    display_name_similarities = [   METRIC_NAMES[ similarity ] if similarity in METRIC_NAMES else similarity
                                        for similarity in ordered_similarities ]
    ax.set(yticks=ind, yticklabels=display_name_similarities)
    plt.title( title )
    
    if any(legend):
        ax.legend(legend, loc='best')
    
    with open('../../docs/stats_viz/{}'.format(file_name), 'wb') as file:
        figure.savefig(file, bbox_inches='tight')
        plt.close(figure)

def plot_time_stats(time_stats, title, file_name):
    figure = plt.figure(figsize=(6, 4), dpi=200)
    ax = figure.add_subplot(1, 1, 1)
    ax.grid(which='major', alpha=0.3)
    time_stats = [ (metric, value / 60) for metric, value in sorted(time_stats.items(), key=lambda x: x[1])]
    metrics = [ METRIC_NAMES[ item[0] ] if item[0] in METRIC_NAMES else item[0] for item in time_stats ]
    time_stats = [ item[1] for item in time_stats ]
    ax.barh( metrics, time_stats, color=get_colors(metrics))
    plt.title( title )
    with open('../../docs/stats_viz/{}'.format( file_name ), 'wb') as file:
        figure.savefig(file, bbox_inches='tight')
        plt.close(figure)

def plot_stats(stats, experiments):

    def plot(data, title, file_name):
        figure = plt.figure(figsize=(6, 4), dpi=200)
        ax = figure.add_subplot(1, 1, 1)
        ax.grid(which='major', alpha=0.3)        
        metrics = [ METRIC_NAMES[ item[0] ] if item[0] in METRIC_NAMES else item[0] for item in data ]    
        data = [ item[1] for item in data ]    
        ax.barh( metrics, data, color=get_colors(metrics))
        plt.title( title )    
        with open('../../docs/stats_viz/{}'.format( file_name ), 'wb') as file:
            figure.savefig(file, bbox_inches='tight')
            plt.close(figure)

    angles_data = [ (metric, np.abs(value["good_rotation_angles"][0])) for metric, value in sorted(stats.items(), key=lambda x: np.abs(x[1]["good_rotation_angles"][0]))]
    plot(angles_data, "Mean of rotation absolute values of good set. Experiments: {}".format( experiments ), "good_rotation_angles_{}.png".format( ",".join(map(str, experiments)) ) )

    angles_data = [ (metric, np.abs(value["bad_rotation_angles"][0])) for metric, value in sorted(stats.items(), key=lambda x: np.abs(x[1]["bad_rotation_angles"][0]))]
    plot(angles_data, "Mean of rotation absolute values of bad set. Experiments: {}".format( experiments ), "bad_rotation_angles_{}.png".format( ",".join(map(str, experiments)) ) )

    sf_data = [ (metric, value["good_scale_factors"][0]) for metric, value in sorted(stats.items(), key=lambda x: x[1]["good_scale_factors"][0])]
    plot(sf_data, "Mean of scale factors of good set. Experiments: {}".format( experiments ), "good_scale_factors_{}.png".format( ",".join(map(str, experiments)) ) )

    sf_data = [ (metric, value["bad_scale_factors"][0]) for metric, value in sorted(stats.items(), key=lambda x: x[1]["bad_scale_factors"][0])]
    plot(sf_data, "Mean of scale factors of bad set. Experiments: {}".format( experiments ), "bad_scale_factors_{}.png".format( ",".join(map(str, experiments)) ) )

    if "good_bg_hgts_mean" in list(stats.values())[0]:
        bg_hgt_data = [ (metric, value["good_bg_hgts_mean"]) for metric, value in sorted(stats.items(), key=lambda x: x[1]["good_bg_hgts_mean"])]
        plot(bg_hgt_data, "Mean of resized BG height of good set. Experiments: {}".format( experiments ), "good_bg_hgts_{}.png".format( ",".join(map(str, experiments)) ) )

        bg_hgt_data = [ (metric, value["bad_bg_hgts_mean"]) for metric, value in sorted(stats.items(), key=lambda x: x[1]["bad_bg_hgts_mean"])]
        plot(bg_hgt_data, "Mean of resized BG height of bad set. Experiments: {}".format( experiments ), "bad_bg_hgts_{}.png".format( ",".join(map(str, experiments)) ) )
    
def plot_resnet_layers_stats(resnet_layers_stats, title, xlabel, ylabel, file_name):
    figure = plt.figure(figsize=(6, 4), dpi=200)
    ax = figure.add_subplot(1, 1, 1) 
    ax.yaxis.set_major_formatter( ticker.PercentFormatter(1.0) )   
    ax.grid(which='major', alpha=0.3)

    coef = 1. # 1.84

    layers = sorted([ layer for layer in resnet_layers_stats ])
    rates = [ resnet_layers_stats[layer] * coef for layer in layers ]
    layers = [ layer.replace("ResNet ", "") for layer in layers ]
        
    #rates = [ rates[0], rates[2], rates[5], rates[7] ]
    #layers = [ layers[0], layers[2], layers[5], layers[7] ]
    
    ax.plot(layers, rates, 'go--', lw=3, markerfacecolor='white', markersize=8)
    plt.xticks(rotation=45)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    with open('../../docs/stats_viz/{}'.format(file_name), 'wb') as file:
        figure.savefig(file, bbox_inches='tight')
        plt.close(figure)    

def plot_improvement_factors( factors, title, file_name ):
    figure = plt.figure(figsize=(6, 4), dpi=200)
    ax = figure.add_subplot(1, 1, 1)
    ax.grid(which='major', alpha=0.3)

    factors = [ (metric, factor) for metric, factor in sorted(factors.items(), key=lambda x: x[1])] #, reverse=True
    metrics = [ factor[0] for factor in factors ]    
    factors = [ factor[1] for factor in factors ]
    ax.barh( metrics, factors, color=get_colors(metrics))
    plt.title( title )
    
    with open('../../docs/stats_viz/{}'.format(file_name), 'wb') as file:
        figure.savefig(file, bbox_inches='tight')
        plt.close(figure)
# ==================== PLOT ENDED ============================


if __name__ == '__main__':
    
    #get_stats_batch( [13], use_only_good_folder = [False] )
    #first_step = get_stats_batch( [15], use_only_good_folder = [False] )
    #get_stats_batch( [13, 15], use_only_good_folder = [False, False] )
    #
    #random_local_metrics = get_stats_batch( [20, 21], use_only_good_folder = [True, False] )    
    #aligned_local_metrics = get_stats_batch( [22], use_only_good_folder = [False] )
    #plot_similarity_double_rates( [random_local_metrics, aligned_local_metrics], "Global search. Random local metrics vs. aligned ones", "global_similarity_rates_random_local_vs_aligned.png", legend=["random", "aligned"] )
    
    #mixed = get_stats_batch( [20, 21, 22], use_only_good_folder = [True, False, False] )
    #improvement_factor = np.mean( list(tuned_search_stats(first_step, random_local_metrics).values()) )    
    #
    #resnet_results([24], use_only_good_folder = [False])
    #resnet_results([25], use_only_good_folder = [False])
    #resnet_results([24, 25], use_only_good_folder = [False, False])
    resnet_results([24, 25, 26], use_only_good_folder = [False, False, False])



    

    
        
    

    
    


