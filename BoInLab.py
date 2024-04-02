import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from matplotlib.colors import CenteredNorm

class BoInLab():

    def __init__(self, data_file, idx_col_name = "rxn_id", obs_col_name = "Y", 
                 sel_descriptor = "default", 
                 rm_zero_by = 0, 
                 set_pbound_mode = "tight", buffer_percent=0, pbound_for={},
                 save_meta = True, preffix = "test"):
        """
        Load data file, column Y is the target or observable
        Columns after Y are descriptors
        Anything in column Y and descriptors columns must be number
        sel_descriptor: list of descriptor name, for example, ["D"]
        by default, all descriptor will be selected
        
        by default, data with Y equals zero will be removed.
        user can specify invalid data to be removed, for example: rm_zero_by = -1, then all Y values equals -1 will be removed.
        if all data needs to be kept, set rm_zero_by as None. 

        check _get_pbound for detials on set_pbound_mode, buffer_percent and pbound_for
        """
        self.obs_name = obs_col_name
        self.origin_data = pd.read_csv(data_file, index_col = idx_col_name)
        self.obs_series = self._get_target(self.origin_data, obs_col_name)
        self._get_descriptors(self.origin_data, obs_col_name, sel_descriptor=sel_descriptor)
        self._rm_zero(rm_zero_by)
        self.obs_stats = self._get_stats(self.obs_series)
        self.descriptor_stats = self._get_stats(self.descriptor_df)
        self._get_pbound(set_pbound_mode, buffer_percent, pbound_for)

        self.save_meta = save_meta

        if save_meta:

            cwd = os.getcwd()
            if not os.path.isdir(preffix):
                os.mkdir(preffix)
                print(f"Create folder: {preffix}")
            
            self.result_folder = cwd + '\\' + preffix
                # print(result_folder)


    

    
    #suppose we want to register data one by one, we should not know the standardised data before the optimisation starts
    def _get_target(self, full_ip_df, target_col_name):
        """
        find the target series from full dataframe
        """
        target_series = full_ip_df[target_col_name]
        return target_series

    
    def _rm_zero(self, rm_zero_by):
        """
        clean data by removing observable equals to rm_zero_by
        """
        if rm_zero_by != None:
            non_zero_obs_series = self.obs_series[self.obs_series != rm_zero_by]
            self.obs_series = non_zero_obs_series

            non_zero_dscpt_df = self.descriptor_df.loc[self.obs_series.index, :]
            self.descriptor_df = non_zero_dscpt_df


    def _get_descriptors(self, full_ip_df, obs_col_name, sel_descriptor):
        """
        first get descriptor col name
        then get the values from full_ip_df
        """
        self._get_descriptors_name(full_ip_df, obs_col_name, sel_descriptor)
        descriptor_df = pd.DataFrame()
        descriptor_df[self.descriptor_col_name] = full_ip_df[self.descriptor_col_name]
        self.descriptor_df = descriptor_df


    def _get_descriptors_name(self, full_ip_df, obs_col_name, sel_descriptor = "default"):
        """
        extract the descriptor column name from full dataframe
        descriptor columns are all columns after target_col
        by default, all descriptors will be picked out, unlecess specify the sel_descriptor argument
        sel_descriptor: list of descriptor name, for example, ["D"]
        """
        for i, col_name in enumerate(list(full_ip_df.columns)):
            if col_name.lower() == obs_col_name.lower():
                all_descpt_name = []
                for n in range(i+1, len(list(full_ip_df.columns))):
                    all_descpt_name.append(list(full_ip_df.columns)[n])
                    self.all_descpt_col_name = all_descpt_name
                break

        if sel_descriptor == "default":
            self.descriptor_col_name = self.all_descpt_col_name
            
        else:
            #check if everything in sel_descriptor is in the full descriptor list

            if set(sel_descriptor).issubset(set(all_descpt_name)):
                self.descriptor_col_name  = sel_descriptor
            else:
                raise ValueError("Unidentified descriptor in sel_descriptor argument.")
    
    def _get_stats(self, ip_df):
        """
        get statistics details
        """
        stat_df = ip_df.describe()
        return stat_df


    def _get_pbound(self, set_pbound_mode = "tight", buffer_percent=0, pbound_for={}, **kwargs):
        """
        set pbound
        
        set_pbound_mode:
        - default is "tight", meaning min and max of current descriptor set are set as pbound
        
        - if specified as "smooth", a buffer region will be applied on the descriptor set;
        the size of buffer zone by default is p/m 10% from current max and min;
        the size of buffer zone can be set in kwargs using "buffer_percent".

        - if specified as "fix_for_selected", the pbound of all descriptor needs to be claimed using "pbound_for" kwarg.
        The rest will be set as "smooth".

        kwargs:
        buffer_percent: float between 0 to 1, specify the buffer bound deviated from current dataset.
        If 0, it's same as "tight"
        
        pbound_for: dict, key is descriptor name, value is set pbound as a tuple.
        For example: {"D":(8, 18), "H":(1, 30)}.

        return:
            pbounds [dict] with signature of {descriptor_name: (pmin, pmax)}
        """
        #read descriptor_stats and get min, max
        all_min_df = self.descriptor_stats.loc['min']
        all_max_df = self.descriptor_stats.loc['max']

        if set_pbound_mode == 'tight':
            #organise the full info into BO package requried format {descriptor_name: (pmin, pmax)}
            all_pbound_dict = self._org_pbound(all_min_df, all_max_df)
            # print(all_pbound_dict)
            self.all_pbound = all_pbound_dict
            
        elif set_pbound_mode == 'smooth':
            smooth_min_df, smooth_max_df = self._smooth_bound(self.descriptor_df, buffer_percent)
            # print(smooth_min_df)
            # print(smooth_max_df)
            all_pbound_dict = self._org_pbound(smooth_min_df, smooth_max_df)
            self.all_pbound = all_pbound_dict
            
        elif set_pbound_mode == 'fix_for_selected':
            if len(pbound_for) == 0:
                raise ValueError("Please specify which descriptors to be fixed using kwarg pbound_for")
            else:
                # print(pbound_for.keys())
                # print(set(pbound_for.keys()))
                # print(set(self.all_descpt_col_name))
                if not set(pbound_for.keys()).issubset(set(self.all_descpt_col_name)):
                    raise ValueError("Unidentified descriptors.")
                else:
                    all_pbound_dict = {}
                    all_pbound_dict.update(pbound_for)
                    # print(all_pbound_dict)
                    rest_descriptor_set = set(self.descriptor_df.columns) - set(pbound_for.keys())
                    # print(rest_descriptor_set)
                    rest_descriptor_df = pd.DataFrame()
                    rest_descriptor_df[list(rest_descriptor_set)] = self.descriptor_df[list(rest_descriptor_set)]
                    rest_smooth_min_df, rest_smooth_max_df = self._smooth_bound(rest_descriptor_df, buffer_percent)
                    
                    rest_pbound_dict = self._org_pbound(rest_smooth_min_df, rest_smooth_max_df)
                    # print(rest_pbound_dict)
                    all_pbound_dict.update(rest_pbound_dict)
                    # print(pbound_for)
                    # print(all_pbound_dict)
                    self.all_pbound = all_pbound_dict
            
                    

        else:
            raise ValueError(f"Methods {set_pbound_mode} not found!")
        #self.obs_series
        #self.descriptor_df
        #have both been defined before this function is called
        print("pbound for all descriptors:\n")
        print(self.all_pbound)

    
    def _org_pbound(self, min_df, max_df):
        """
        convert min_df and max_df into a dict wiht signature of {descriptor_name: (min, max)}
        """
        min_dict = min_df.to_dict()
        max_dict = max_df.to_dict()
        pbound_dict = {}
        for descpt in min_dict.keys():
            pbound_dict[descpt] = (min_dict[descpt], max_dict[descpt])
        
        return pbound_dict

    # def _smooth_bound_by_percentile(self, full_df, buffer_percent):
    #     """
    #     this function is optional, the distribution of data is considered by percentile
    #     """
    #     smooth_min_df = full_df.min() - (full_df.quantile(q = buffer_percent) - full_df.min())
    #     smooth_max_df = full_df.max() + (full_df.max() - full_df.quantile(q = 1- buffer_percent))
    #     return smooth_min_df, smooth_max_df

    def _smooth_bound(self, full_df, buffer_percent):
        span_df = full_df.max() - full_df.min()
        buffer_df = span_df * buffer_percent
        smooth_min_df = full_df.min() - buffer_df
        smooth_max_df = full_df.max() + buffer_df
        return smooth_min_df, smooth_max_df
    

    def init_bayesian(self, alpha=0.5, verbose = 2, randstate = 1, 
                      acq_func = "ucb", ucb_kappa = 0, xi = 1e-2, 
                      ):
        """
        init bayesian optimisation
        For gaussian process, specify noise level using "alpha"
        verbose and randstate please refer to BayesianOptimization package doc

        default acq_func is "ucb", other options are "poi" and "ei"
        for "ucb", the required parameter is ucb_kappa, default 0
        for ei and poi, the required parameter is xi, default 1e-2, obs_max will be extracted during calculation
        """
        self.optimizer = BayesianOptimization(
            f = None,
            pbounds = self.all_pbound,
            verbose = verbose,
            random_state = randstate,
        )
        self.optimizer.set_gp_params(alpha = alpha)
        self.register_data()
        self.acq(acq_func, ucb_kappa, xi)

        if self.save_meta:
            op_filename = f"{self.result_folder}\\res_{acq_func}.txt"
            with open(op_filename, "a") as f:
                f.write("\n\n=========================================\n")
                f.write("Input parameters:\n")
                f.write(f"{alpha=}\n")
                f.write(f"{acq_func=}\n")
                if acq_func == "ucb":
                    f.write(f"{ucb_kappa=}\n")
                else:
                    f.write(f"{xi=}\n")
                

                f.write("---------------------\n")
                f.write("Next point to go:\n")
                f.write(f"{self.res}")
                f.write("===========================================\n\n")

    
    def register_data(self):
        """
        add data from self.descriptor_df and self.obs_series one by one
        data from descriptor_df go to params
        data from obs_seris go to target
        """
        for i, this_obs in enumerate(self.obs_series):
            target = this_obs
            this_descpt_row = self.descriptor_df.iloc[i]
            params = {}
            for descpt in self.descriptor_df.columns:
                params[descpt] = this_descpt_row[descpt]
            
            # print(params)
            # print(target)
            print(f"Register {this_descpt_row.name}...")
            self.optimizer.register(params = params, target=target)
            print(f"Done.")
    
    def acq(self, acq_func = "ucb", ucb_kappa = 0, xi = 1e-2):
        #init acq_func
        if acq_func.lower() not in ["ucb", "poi", "ei"]:
            raise ValueError("Unidentified acquisition function. Please select from ['ucb', 'poi', 'ei'].")
        else:
            acq_kind = acq_func.lower()
            print(acq_kind)
            
            if acq_kind == "ucb":
                self.utility = UtilityFunction(kind = "ucb", kappa = ucb_kappa)
                self.ucb_kappa = ucb_kappa
                print(f"{ucb_kappa=}")
            
            elif acq_kind == "poi":
                self.xi = xi
                self.utility = UtilityFunction(kind = "poi", xi = xi)
                print(f"{xi=}")
            
            elif acq_kind == "ei":
                self.utility = UtilityFunction(kind = "ei", xi = xi)
                self.xi = xi
                print(f"{xi=}")
            self.acq_func = acq_func
            self.res = self.optimizer.suggest(self.utility)
            print("Suggested next point to go:")
            print(self.res)
    

    def plot_res(self, plot_mode = "any_two", two_to_plot = ["P", "H"]):
        """
        plot_mode:
        "any_two" will do a 2d plot with selected descriptor specified in two_to_plot
        "any_single" will do a 2d plot with observable versus selected descriptor 
        """
        if plot_mode == "any_two":
            if not set(two_to_plot).issubset(set(self.all_descpt_col_name)):
                raise ValueError("Unidentified descriptors.")
            elif len(two_to_plot) != 2:
                raise ValueError("Please specify two descriptors using kwarg two_to_plot = [].")
            else:
                res_coord = {}

                for descpt in two_to_plot:
                    res_coord[descpt] = self.res[descpt]
                
                descpt_0 = two_to_plot[0]
                descpt_1 = two_to_plot[1]

                descpt_0_pbound_min = self.all_pbound[descpt_0][0]
                descpt_0_pbound_max = self.all_pbound[descpt_0][1]
                descpt_1_pbound_min = self.all_pbound[descpt_1][0]
                descpt_1_pbound_max = self.all_pbound[descpt_1][1]

                #prep for countour plot
                # X_descpt_0, X_descpt_1 = np.mgrid[descpt_0_pbound_min:descpt_0_pbound_max:101j, descpt_1_pbound_min:descpt_1_pbound_max:101j]
                # xdescpt = [[x_descpt_0, x_descpt_1] for x_descpt_0, x_descpt_1 in zip(X_descpt_0.flatten(), X_descpt_1.flatten())]
                # util_flat = self.utility.utility(xdescpt, self.optimizer._gp, self.optimizer.max['target'])
                # util = util_flat.reshape(101, 101)

                
                # print(res_coord)
                plt.scatter(self.descriptor_df[descpt_0], self.descriptor_df[descpt_1], c = self.obs_series)
                plt.colorbar()
                plt.xlabel(descpt_0)
                plt.ylabel(descpt_1)

                if self.acq_func == "ucb":
                    plt.scatter(res_coord[descpt_0], res_coord[descpt_1], marker = "*", s = 100, color = "r", 
                                label = f"ucb_kappa:{self.ucb_kappa:.2f}\n({res_coord[descpt_0]:.2f}, {res_coord[descpt_1]:.2f})")
                    scatter_plot_name = f"sc_{descpt_0}_{descpt_1}_{self.acq_func}_{self.ucb_kappa:.2f}.png"
                
                plt.legend()
                plt.savefig(f"{self.result_folder}\\{scatter_plot_name}") 
                # #generate plot name and save figure


                

                pass
        pass




    





                

