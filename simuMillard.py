import warnings
import matplotlib as plt
import tellurium as te
warnings.filterwarnings('error')
te.setDefaultPlottingEngine('matplotlib')
plt.interactive(False)


def loadPModel(file):
    r = te.loadSBMLModel(file)
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
    # file = 'E_coli_Millard2016-L3V1.xml'
    # s_prom = loadModelAndPromoteToGlobalParameters(file)
    # # Number of simulations = num_of_random_vmaxes * 22(=len(glc_vector))
    # num_of_random_vmaxes = 100
    # rr = roadrunner.RoadRunner(s_prom)
    # model = rr.model
    # vmax_list = getVmaxIdsAndValues(model)
    # vmax_matrix = createVmaxMatrix(vmax_list, num_of_random_vmaxes)
    # glc_vector = makeGlcVector()
    # results_1 = []
    # # results_2 = []
    # # results_1 = runParamSimu(rr, glc_vector, vmax_matrix, vmax_list)
    # # results_2 = runParamSteadySimu(s_prom, glc_vector, vmax_matrix, vmax_list)
    # # print(results_1)
    # np.savetxt("tmp_Results_Growth.csv", results_1, delimiter=",")
    # create_boxplot(results_1)

    # print loading info
    print('#' * 80)
    print('BioApp version: 0.01')
    print('#' * 80)

    # Main Program
    # Manual Settings  **CARE**
    filename = 'E_coli_Millard2016-L3V1.xml'
    # # Number of simulations = num_of_random_vmaxes * 22(=len(glc_vector))
    # num_of_random_vmaxes = 100
    sbml_prom = loadPModel(filename)  # loadModelAndPromoteToGlobalParameters(filename)
    rr = te.loadSBMLModel(sbml_prom)
    getInfo(rr)
    result = rr.simulate(0, 40, 500)
    rr.plot(result, show=False, loc=None, color='black', alpha=0.7)
    te.show()


if __name__ == '__main__':
    main()
