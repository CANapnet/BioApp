import tellurium as te

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
    print('BioApp v. 0.01')
    print('#' * 80)

    # to get the tellurium version use
    print('te.__version__')
    print(te.__version__)
    # or
    print('te.getTelluriumVersion()')
    print(te.getTelluriumVersion())

    # to print the full version info use
    print('-' * 80)
    te.printVersionInfo()
    print('-' * 80)


if __name__ == '__main__':
    main()
