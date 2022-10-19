import json
import pandas as pd
import numpy as np

# from private_data_generator import PrivateDataGenerator

class PublicationAggregator():
    """
    Central class for aggregator functionality.
    
    Provides methods for:
    1. Generating results across publications comparing real to private
       performance

    2. Generating summary results over all publications (i.e. average 
       citations, etc.)
    
    3. Generating publication level results for single publications (works by
       default, where n_publications = 1)

    TODO: Helper method that runs all findings once for this class
    """

    EPSILONS = [(np.e ** -1, 'e^-1'), 
            (np.e ** 0, 'e^0'), 
            (np.e ** 1, 'e^1'),
            (np.e ** 2, 'e^2')]

    ITERATIONS = 5

    def __init__(self, publications):
        self.publications = publications
        self.findings_map = {}

    def _run_all_findings(self, p, pub_id, str_eps): # data_generator
        if pub_id + '_' + str_eps in self.findings_map.keys():
            return self.findings_map[pub_id + '_' + str_eps]

        folder_name = 'private_data/' + str(pub_id) + '/' + str_eps + '/'

        mst_findings = []
        patectgan_findings = []
        privbayes_findings = []
        for it in range(self.ITERATIONS):
            mst_df = pd.read_pickle(folder_name + 'mst_' + str(it) + '.pickle')
            mst_results = p(dataframe=mst_df).run_all_non_visual_findings()
            mst_findings.append(mst_results)
            
            patectgan_df = pd.read_pickle(folder_name + 'patectgan_' + str(it) + '.pickle')
            patectgan_results = p(dataframe=patectgan_df).run_all_non_visual_findings()
            patectgan_findings.append(patectgan_results)

            privbayes_df = pd.read_pickle(folder_name + 'privbayes_' + str(it) + '.pickle')
            privbayes_results = p(dataframe=privbayes_df).run_all_non_visual_findings()
            privbayes_findings.append(privbayes_results)
        
        self.findings_map[pub_id + '_' + str_eps] = (mst_findings, patectgan_findings, privbayes_findings)
        return self.findings_map[pub_id + '_' + str_eps]

    def real_vs_private_soft(self):
        """
        Calculate percent matching of soft findings on
        real vs private across epsilons and synthesizers
        """
        # pub_id -> percentages_at_epsilon
        soft_percentages = {}

        def synth_helper(synth_name, findings, str_eps, pub_id):
            total_percent = 0
            for synth in findings:
                bool_synth = []
                for _, result in synth.items():
                    bool_synth.append(result[1])
                
                total_percent += sum([1 if bool_synth[i] == real_bool_soft[i] else 0 for i in range(len(bool_synth))]) / len(synth)
                
            total_percent = total_percent / len(findings)
            soft_percentages[pub_id][str_eps][synth_name] = total_percent

        for p in self.publications:
            # epsilon -> percent_soft_findings
            pub_id = p.DEFAULT_PAPER_ATTRIBUTES['id']
            pub_file_base_df = p.DEFAULT_PAPER_ATTRIBUTES['base_dataframe_pickle']
            soft_percentages[pub_id] = {}

            p_base_instantiated = p(filename=pub_file_base_df)
            # data_generator = PrivateDataGenerator(p_base_instantiated)

            # Run all real non visual findings 
            real_results = p_base_instantiated.run_all_non_visual_findings()
            real_bool_soft = []
            for _, result in real_results.items():
                real_bool_soft.append(result[1])

            # In case data has not already been generated
            # data_generator.generate()

            for (_, str_eps) in self.EPSILONS:
                # Create the findings
                mst_findings, patectgan_findings, privbayes_findings = self._run_all_findings(#data_generator,
                                                                                            p,
                                                                                            pub_id,
                                                                                            str_eps)
                soft_percentages[pub_id][str_eps] = {}

                synth_helper('mst', mst_findings, str_eps, pub_id)
                synth_helper('patectgan', patectgan_findings, str_eps, pub_id)
                synth_helper('privbayes', privbayes_findings, str_eps, pub_id)
        
        return soft_percentages

    def real_vs_private_soft_error_bars(self):
        """
        Calculate percent matching of soft findings on
        real vs private across epsilons and synthesizers
        """
        from statsmodels.stats.weightstats import DescrStatsW

        # pub_id -> percentages_at_epsilon
        #soft_means = {}
        #soft_lower_cis = {}
        #soft_upper_cis = {}
        soft_data = {}
        soft_data["str_eps"] = []
        soft_data["synth"] = []
        soft_data["value"] = []

        def synth_helper(synth_name, findings, str_eps, pub_id):
            proportion_array = []
            for synth in findings:
                bool_synth = []
                for _, result in synth.items():
                    bool_synth.append(result[1])
                
                proportion_array.append(sum([1 if bool_synth[i] == real_bool_soft[i] else 0 for i in range(len(bool_synth))]) / len(bool_synth))
                soft_data["str_eps"].append(str_eps)
                soft_data["synth"].append(synth_name)
                soft_data["value"].append(sum([1 if bool_synth[i] == real_bool_soft[i] else 0 for i in range(len(bool_synth))]) / len(bool_synth))
            
            #soft_means[pub_id][str_eps][synth_name] = np.mean(proportion_array)
            #conf_int = DescrStatsW(data=proportion_array, weights=None).tconfint_mean()
            #soft_lower_cis[pub_id][str_eps][synth_name] = conf_int[0]
            #soft_upper_cis[pub_id][str_eps][synth_name] = conf_int[1]
            
        for p in self.publications:
            # epsilon -> percent_soft_findings
            pub_id = p.DEFAULT_PAPER_ATTRIBUTES['id']
            pub_file_base_df = p.DEFAULT_PAPER_ATTRIBUTES['base_dataframe_pickle']
            #soft_means[pub_id] = {}
            #soft_lower_cis[pub_id] = {}
            #soft_upper_cis[pub_id] = {}

            p_base_instantiated = p(filename=pub_file_base_df)
            # data_generator = PrivateDataGenerator(p_base_instantiated)

            # Run all real non visual findings 
            real_results = p_base_instantiated.run_all_non_visual_findings()
            real_bool_soft = []
            for _, result in real_results.items():
                real_bool_soft.append(result[1])

            # In case data has not already been generated
            # data_generator.generate()

            for (_, str_eps) in self.EPSILONS:
                # Create the findings
                mst_findings, patectgan_findings, privbayes_findings = self._run_all_findings(#data_generator,
                                                                                            p,
                                                                                            pub_id,
                                                                                            str_eps)
                #soft_means[pub_id][str_eps] = {}
                #soft_lower_cis[pub_id][str_eps] = {}
                #soft_upper_cis[pub_id][str_eps] = {}

                synth_helper('mst', mst_findings, str_eps, pub_id)
                synth_helper('patectgan', patectgan_findings, str_eps, pub_id)
                synth_helper('privbayes', privbayes_findings, str_eps, pub_id)
        
        return soft_data

    def finding_arrays_soft(self, str_eps):
        """
        Simply return arrays of soft findings for each
        publication for the main figure at an epsilon value
        """
        finding_maps = {}

        def synth_helper(synth_name, findings, pub_id):
            aggregate = np.zeros(len(findings[0]))
            for synth in findings:
                for i, (_, result) in enumerate(synth.items()):
                    if result[1] == real_bool_soft[i]:
                        aggregate[i] += 1
                
            finding_maps[pub_id][synth_name] = aggregate
        
        for p in self.publications:
            # epsilon -> percent_soft_findings
            pub_id = p.DEFAULT_PAPER_ATTRIBUTES['id']
            pub_file_base_df = p.DEFAULT_PAPER_ATTRIBUTES['base_dataframe_pickle']
            finding_maps[pub_id] = {}

            p_base_instantiated = p(filename=pub_file_base_df)
            # data_generator = PrivateDataGenerator(p_base_instantiated)

            # Run all real non visual findings 
            real_results = p_base_instantiated.run_all_non_visual_findings()
            real_bool_soft = []
            for _, result in real_results.items():
                real_bool_soft.append(result[1])

            # In case data has not already been generated
            # data_generator.generate()

            # Create the findings
            mst_findings, patectgan_findings, privbayes_findings = self._run_all_findings(# data_generator,
                                                                                        p,
                                                                                        pub_id,
                                                                                        str_eps)

            synth_helper('mst', mst_findings, pub_id)
            synth_helper('patectgan', patectgan_findings, pub_id)
            synth_helper('privbayes', privbayes_findings, pub_id)
        
        return finding_maps




                