

from configs.config import *
from utilities import *
import time

import os, glob, scipy.io

# Set up paths
path = os.getcwd()
fixed_threshold= 0.5
specific_experiment = "Real World Datasets"
Alg_Group = 'All' # All /  Fast / Slow / GPU
Algorithm = ''
specific_experiment = "feature_selection_benchmark"  #+"_"+Alg_Group+"_"+Algorithm


# Retrieve a sorted list of algorithms
algorithms = get_sorted_algorithms(Alg_Group)
#algorithms = ['ETree', "AdaBoost"] 
# Get Real World Datasets: 
unique_folders = get_unique_folder_names(data_path)

for algo_info in algorithms:
    algo_name = algo_info['name']
    # enable use to run only specific algorithms maually
    
    if fs_algorithms_list !='All' :
        if algo_name not in ['ETree'] :
            continue
    alg = algo_info['function']
    hyperparams = algo_info['hyper']
    
    for repository in unique_folders:
        repository_path = os.path.join(data_path, repository)
        data_path_loop = repository_path
        data_path_temp = os.path.join(data_path_loop, '*.mat')
        datasets = glob.glob(data_path_temp)
        
        for dataset_path in datasets:
            data_name = remove_mat_extension(os.path.relpath(dataset_path, data_path_loop))
            #if "Peeling" in data_name or "_Test" in data_name:
            #    continue
            if data_name not in ['MCI_data'] :# Period Changer','slashdot','tumors C','Breast','hepatitis','colon','HIVA','SMK-CAN-187','Titanic'] : #,'low_variance','ETree', 'SVC','LS','MCFS','DT', 'Skewness', 'AdaBoost','alpha_investing', 'SHAP'] :
                continue 
            Data = scipy.io.loadmat(dataset_path)
            data_arr = Data['X']
            label_arr = Data['Y'][:, 0] if min(Data['Y'][:, 0]) == 0 else Data['Y'][:, 0] - 1
            rows, columns_len = data_arr.shape
            df_FS, columns = read_or_create_result_table(path_output_excel_FS,specific_experiment)
            
            filtered_k_fs, filtered_k_hd = check_k_limitation_and_run(algo_name, hyperparams, df_FS, data_name, rows, columns_len,fixed_threshold,perform_cv,initial_splits,train_size,evaluation_models,no_hyperparameters,K_List)
            # Only for ETree and AdaBoost we apply HD experiment :
            start_time = time.time()

            #process_fs_algorithm(label_arr,specific_experiment, fixed_threshold,data_name, algo_name, filtered_k_fs, dataset_path,hyperparams, repository, columns_len, path_output_excel_FS , alg, columns, 'No FS',debug)
    
            if algo_name in ['ETree', 'AdaBoost'] :
                if not filtered_k_fs and filtered_k_hd == 'Y':
                    print(f"For Dataset: {data_name}, all K values already run for algorithm: {algo_name}")
                    continue
                else:
                    if not filtered_k_fs and filtered_k_hd =='N' : # Not run HD yet but run FS (all K  values)
                        print("For Dataset : " + data_name + " All K values already run for algorithm: " + algo_name + " Not run High Dimentional dataset and Experiment Type: " + Experimnet_Type)
                        from datetime import datetime; start_time = datetime.now(); print("Start time:", start_time.strftime("%H:%M"), "| Day:", start_time.strftime("%A, %d %B %Y"))                
                        process_fs_algorithm(label_arr,specific_experiment, fixed_threshold,data_name, algo_name, filtered_k_fs, dataset_path,hyperparams, repository, columns_len, path_output_excel_FS , alg, columns, 'No FS',debug)
                    else :

                        if filtered_k_hd == 'Y': # Not run FS - run HD
                            print("For Dataset : " + data_name + " Partial run for K values already run for algorithm: " + algo_name + " For High Dimentional dataset and Experiment Type: " + Experimnet_Type)    

                            from datetime import datetime; start_time = datetime.now(); print("Start time:", start_time.strftime("%H:%M"), "| Day:", start_time.strftime("%A, %d %B %Y"))
                            
                            process_fs_algorithm(label_arr,specific_experiment, fixed_threshold,data_name, algo_name, filtered_k_fs, dataset_path,hyperparams, repository, columns_len, path_output_excel_FS , alg, columns, 'FS',debug)

                        else :
                            from datetime import datetime; start_time = datetime.now(); print("Start time:", start_time.strftime("%H:%M"), "| Day:", start_time.strftime("%A, %d %B %Y"))
                            process_fs_algorithm(label_arr,specific_experiment, fixed_threshold, data_name, algo_name, filtered_k_fs, dataset_path,hyperparams, repository, columns_len, path_output_excel_FS , alg, columns, 'No FS',debug)

                            from datetime import datetime; start_time = datetime.now(); print("Start time:", start_time.strftime("%H:%M"), "| Day:", start_time.strftime("%A, %d %B %Y"))
                            process_fs_algorithm(label_arr,specific_experiment, fixed_threshold,data_name, algo_name, filtered_k_fs, dataset_path,hyperparams, repository, columns_len, path_output_excel_FS , alg, columns, 'FS',debug)

            else :
                if not filtered_k_fs :
                    print("For Dataset : " + data_name + " All K values already run for algorithm: " + algo_name)
                    continue
                else :

                    from datetime import datetime; start_time = datetime.now(); print("Start time:", start_time.strftime("%H:%M"), "| Day:", start_time.strftime("%A, %d %B %Y"))
                    process_fs_algorithm(label_arr,specific_experiment, fixed_threshold, data_name, algo_name, filtered_k_fs, dataset_path,hyperparams, repository, columns_len, path_output_excel_FS , alg, columns, 'FS',debug)


