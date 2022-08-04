import json
import pandas as pd

from private_data_generator import PrivateDataGenerator

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
    """

    def __init__(self, publications):
        self.publications = publications

    def real_vs_private_soft(self):
        """
        Calculate percent matching of soft findings on
        real vs private across epsilons and synthesizers
        """
        # pub_id -> percentages_at_epsilon
        soft_percentages = {}
        for p in self.publications:
            # epsilon -> percent_soft_findings
            pub_id = p.DEFAULT_PAPER_ATTRIBUTES['id']
            pub_file_base_df = p.DEFAULT_PAPER_ATTRIBUTES['base_dataframe_pickle']
            soft_percentages[pub_id] = {}

            p_base_instantiated = p(filename=pub_file_base_df)
            data_generator = PrivateDataGenerator(p_base_instantiated)

            # Run all real non visual findings 
            real_results = p_base_instantiated.run_all_non_visual_findings()
            real_bool_soft = []
            for _, result in real_results.items():
                real_bool_soft.append(result[1])

            # In case data has not already been generated
            data_generator.generate()
            for (_, str_eps) in data_generator.EPSILONS:
                soft_percentages[pub_id][str_eps] = {}
                folder_name = 'private_data/' + str(pub_id) + '/' + str_eps + '/'

                mst_findings = []
                patectgan_findings = []
                for it in range(data_generator.ITERATIONS):
                    mst_df = pd.read_pickle(folder_name + 'mst_' + str(it) + '.pickle')
                    mst_results = p(dataframe=mst_df).run_all_non_visual_findings()
                    mst_findings.append(mst_results)
                    
                    patectgan_df = pd.read_pickle(folder_name + 'patectgan_' + str(it) + '.pickle')
                    patectgan_results = p(dataframe=patectgan_df).run_all_non_visual_findings()
                    patectgan_findings.append(patectgan_results)

                total_percent_mst = 0
                for mst in mst_findings:
                    bool_mst = []
                    for _, result in mst.items():
                        bool_mst.append(result[1])
                    
                    total_percent_mst += sum([1 if bool_mst[i] == real_bool_soft[i] else 0 for i in range(len(bool_mst))]) / len(mst)
                    
                total_percent_mst = total_percent_mst / len(mst_findings)
                soft_percentages[pub_id][str_eps]['mst'] = total_percent_mst

                total_percent_patectgan = 0
                for patectgan in patectgan_findings:
                    bool_patectgan = []
                    for _, result in patectgan.items():
                        bool_patectgan.append(result[1])
                    
                    total_percent_patectgan += sum([1 if bool_patectgan[i] == real_bool_soft[i] else 0 for i in range(len(bool_patectgan))]) / len(patectgan)
                    
                total_percent_patectgan = total_percent_patectgan / len(patectgan_findings)
                soft_percentages[pub_id][str_eps]['patectgan'] = total_percent_patectgan
        
        return soft_percentages



                