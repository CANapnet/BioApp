import re
import warnings
import roadrunner as rr
import numpy as np
import progressbar
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

warnings.filterwarnings('error')


# te.setDefaultPlottingEngine('matplotlib')
# plt.interactive(True)
# mpl.interactive(True)


def create_bx(m_growth, m_error, glc_vector):
    ##df1 = pd.read_csv("tmp_Results_Growth2.csv", index_col=0)
    #                   names=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
    #                          0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 0.23])
    # df2 = pd.DataFrame(df1, columns=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    #                                  0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 0.23])

    # df1 = pd.DataFrame(m_growth, columns=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    #                                       0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 0.23])
    # df2 = pd.DataFrame(m_error, columns=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
    #                                      0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 0.23])

    # df1.to_csv('Growth_M.csv')
    # df2.to_csv('Error_M.csv')

    df1 = pd.read_csv('Growth_M.csv', index_col=0)
    df2 = pd.read_csv('Error_M.csv', index_col=0)

    # fig, axes = plt.subplots(nrows=2, ncols=2)
    # df2['freq'] = df2.groupby('')['0.0'].transform('count')
    errors = [1, 2, 3, 4]
    zero_freq = np.zeros(shape=(4, 22), dtype=int)
    df3 = pd.DataFrame(zero_freq, columns=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                                           0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 0.23])
    df4 = pd.DataFrame(0, index=np.arange(1, 5), columns=[0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                                                          0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
                                                          1.00, 0.23])
    # print(df3)
    data_freq = df2.apply(pd.value_counts).fillna(0)
    df5 = pd.DataFrame(data_freq)
    # print(df5)

    # for column in df2.columns[0:]:
    #     # counts1 = df2[column].value_counts().to_dict()
    #     # counts2 = df2[column].value_counts().reset_index()
    #     # counts3 = df2[column].value_counts('2')
    #     # df4 = pd.crosstab(df2[index], df2[column])
    #     tria = df2[column].value_counts().index('3')
    #     print(tria)

    # df2.apply(pd.value_counts).fillna(0)
    # df2.Color.value_counts().reset_index().rename(columns={'index': 'Color', 0: 'count'})

    df1.boxplot()
    plt.xlabel('Glc FEED')
    plt.ylabel('Growth rate h-1')
    plt.savefig('Boxplot_GrowthRate.png')
    # df5.plot.bar(ax=axes[1, 0])
    df6 = df5.T
    df6.plot.bar()
    plt.xlabel('Glc FEED')
    plt.ylabel('Simulations')
    plt.show()
    plt.savefig('Counts_Errors.png')
    # print(df1)
    # print(df2)

    # nan_values = df1.isnull().sum()
    # print(nan_values)
    # # plt.boxplot(df1)
    # plt.bar(nan_values)
    # plt.show()


def create_boxplot():
    df1 = pd.read_csv("tmp_Results_Growth2.csv", index_col=0)

    nan_values = df1.isnull().sum()
    print(nan_values)
    print(df1)


def runParamSimu(sbml_prom, glc_vector, vmax_matrix, vmax_list):
    vlist = np.asarray(vmax_list)
    vmax_ids = vlist[:, 0]
    bar = progressB(len(glc_vector) * vmax_matrix.shape[0])
    bar_i = 0
    list_result = []
    matrix_growth_results = np.zeros((vmax_matrix.shape[0], len(glc_vector)), dtype=float)
    matrix_errors = np.zeros((vmax_matrix.shape[0], len(glc_vector)), dtype=int)
    for idx_glc, tmp_glc_value in enumerate(glc_vector):
        for idx_vmax, tmp_vmax_vector in enumerate(vmax_matrix):
            r1 = rr.RoadRunner(sbml_prom)
            # rr.resetAll()
            # rr.reset()
            # print("BEFORE :\tFEED:\t", r1.model['FEED'], "\tGLC_feed:\t", r1.model['GLC_feed'], "\tRPI VMAX:\t",
            #       r1.model['RPI_Vmax'])
            r1.model['FEED'] = tmp_glc_value
            for i, tmp_vmax_id in enumerate(vmax_ids):
                r1.model[tmp_vmax_id] = tmp_vmax_vector[i]
            time_0 = 0
            time_f = 200 * 3600
            num_points = 1000
            # print("AFTER :\tFEED:\t", r1.model['FEED'], "\tGLC_feed:\t", r1.model['GLC_feed'], "\tRPI VMAX:\t",
            #       r1.model['RPI_Vmax'])
            r1.getIntegrator().relative_tolerance = 1e-8
            r1.getIntegrator().absolute_tolerance = 1e-10
            bar_i = bar_i + 1
            bar.update(bar_i)
            try:
                growth_idx = r1.model.getReactionIds().index("GROWTH")
                glc_feed_idx = r1.model.getReactionIds().index("GLC_feed")
                glc_xch_idx = r1.model.getReactionIds().index("XCH_GLC")
                results1 = r1.simulate(time_0, time_f, num_points)
                growth_rate = r1.model.getReactionRates()[growth_idx]
                glc_feed_rate = r1.model.getReactionRates()[glc_feed_idx]
                glc_xch_rate = r1.model.getReactionRates()[glc_xch_idx]
                check_range = check_range_of_rates(glc_feed_rate, glc_xch_rate)
                check_growth = check_gr(growth_rate)
                if check_range == 1:
                    if check_growth == 1:
                        results2 = growth_rate * 3600
                        list_result.append(results1)
                        matrix_growth_results[idx_vmax, idx_glc] = results2
                        matrix_errors[idx_vmax, idx_glc] = 1
                    else:
                        matrix_growth_results[idx_vmax, idx_glc] = np.nan
                        matrix_errors[idx_vmax, idx_glc] = 2
                else:
                    list_result.append(np.nan)
                    matrix_growth_results[idx_vmax, idx_glc] = np.nan
                    matrix_errors[idx_vmax, idx_glc] = 3

            except(RuntimeError, TypeError, NameError, Warning):
                list_result.append(np.nan)
                matrix_growth_results[idx_vmax, idx_glc] = np.nan
                matrix_errors[idx_vmax, idx_glc] = 4
                continue
    return matrix_growth_results, matrix_errors


def progressB(value):
    with progressbar.ProgressBar(max_value=value) as bar:
        return bar


def check_gr(growth):
    growth = growth * 3600
    check = 0
    if growth > 0.01:
        check = 1
        return check
    else:
        return check


def check_range_of_rates(feed, xch):
    a = (abs(feed - xch) / xch) * 100
    b = (abs(xch - feed) / feed) * 100
    check = 0
    if a < 5 or b < 5:
        check = 1
        return check
    else:
        return check


def makeGlcVector():
    glc_vector = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
                  0.85, 0.90, 0.95, 1.00, 0.23]
    # glc_vector_reverse = glc_vector[::-1]
    return glc_vector


def createVmaxMatrix(vmax_list, num_of_random_vmaxes):
    np.random.seed(19680801)
    vlist = np.asanyarray(vmax_list)
    vmax_values = vlist[:, 1]
    vmax_min = np.log((0.8 * np.array(vmax_values.astype(float))))
    vmax_max = np.log((2 * np.array(vmax_values.astype(float))))
    random_log_vmax = np.random.uniform(vmax_min, vmax_max, (num_of_random_vmaxes, len(vlist)))
    random_vmax_matrix = np.exp(random_log_vmax)
    matrix = np.asanyarray(random_vmax_matrix)
    return matrix


def getVmaxIdsAndValues(model):
    all_ids = list(model.items())
    vmax_ids_values = []
    vmax_idx = []
    regexobject_vmax = re.compile('(Vmax)')
    regexobject_init = re.compile('(?![init])')
    for idx, tmp_Vmax_id in enumerate(all_ids):
        result = regexobject_vmax.search(tmp_Vmax_id[0])
        if result:
            result2 = regexobject_init.match(tmp_Vmax_id[0])
            if result2:
                vmax_ids_values.append(tmp_Vmax_id)
                vmax_idx.append(idx)
    return vmax_ids_values


def loadPModel(file):
    r = rr.RoadRunner(file)
    s_orig = r.getSBML()
    s_prom = r.getParamPromotedSBML(s_orig)
    # rr.load(p_sbml)
    # rr.conservedMoietyAnalysis = True
    return s_prom


def getInfo(r):
    num_species = r.model.getNumFloatingSpecies()
    num_reactions = r.model.getNumReactions()
    num_bounds = r.model.getNumBoundarySpecies()
    num_comp = r.model.getNumCompartments()
    num_cm = r.model.getNumConservedMoieties()
    num_float_sp = r.model.getNumFloatingSpecies()
    num_global_par = r.model.getNumGlobalParameters()
    glc_feed = r.FEED
    list_info = [num_species,
                 num_reactions,
                 num_bounds,
                 num_comp,
                 num_cm,
                 num_float_sp,
                 glc_feed]

    print("Number of species:\t{}\n"
          "Number of reactions:\t{}\n"
          "Number of bounds:\t{}\n"
          "Number of compartments:\t{}\n"
          "Number of Conserved Moieties:\t{}\n"
          "Number of floating species:\t{}\n"
          "Number of global parameters:\t{}\n"
          "Glc FEED value:\t{}\n".format(num_species,
                                         num_reactions,
                                         num_bounds,
                                         num_comp,
                                         num_cm,
                                         num_float_sp,
                                         num_global_par,
                                         glc_feed))

    return list_info


def main():
    # print loading info
    print('#' * 80)
    print('BioApp version: 0.01')
    print('#' * 80)

    # Main Program
    # Manual Settings  **CARE**
    filename = 'E_coli_Millard2016-L3V1.xml'
    # # Number of simulations = num_of_random_vmaxes * 22(=len(glc_vector))
    num_of_random_vmaxes = 100

    sbml_prom = loadPModel(filename)  # loadModelAndPromoteToGlobalParameters(filename)
    r1 = rr.RoadRunner(sbml_prom)
    model = r1.model
    vmax_list = getVmaxIdsAndValues(model)
    vmax_matrix = createVmaxMatrix(vmax_list, num_of_random_vmaxes)
    glc_vector = makeGlcVector()

    # results1, results2 = runParamSimu(sbml_prom, glc_vector, vmax_matrix, vmax_list)
    results1 = []
    results2 = []
    create_bx(results1, results2, glc_vector)


if __name__ == '__main__':
    main()
