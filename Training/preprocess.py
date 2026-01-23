import sys, getopt, os
# Data processing and analysis libraries
import pandas as pd
import numpy as np
import json
from scipy import stats
from pandas.api.types import is_numeric_dtype
# Machine learning utilities
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
# Flow cytometry data reader
import flowio


def _validate_training_config(training_cfg):
    """
    Validate that all required training configuration keys are present
    Args:
        training_cfg: Dictionary containing training parameters
    Raises:
        ValueError: If training_cfg is not a dictionary
        KeyError: If required configuration keys are missing
    """
    required_keys = [
        "seed",                          # Random seed for reproducibility
        "fsc_limit",                     # Min FSC threshold for plastic standards
        "fsc_limit_max",                 # Max FSC threshold for plastic standards
        "ssc_limit",                     # Min SSC threshold for plastic standards
        "ssc_limit_max",                 # Max SSC threshold for plastic standards
        "fsc_limit_control",             # Min FSC threshold for control samples
        "ssc_limit_control",             # Min SSC threshold for control samples
        "out_of_scale",                  # Whether to exclude out-of-scale values
        "max_value",                     # Maximum allowed measurement value
        "gating_control_values",         # Whether to gate control samples
        "max_number_nmp_type",           # Max samples per non-microplastic type
        "max_number_standards_type",     # Max samples per plastic standard type
        "min_number_standards_type",     # Min samples required per type
        "outlier_zscore",                # Z-score threshold for outlier removal
        "standards_folder",              # Path to plastic standards folder
    ]
    if not isinstance(training_cfg, dict):
        raise ValueError("model_config['training'] must be a mapping")
    missing = [key for key in required_keys if key not in training_cfg]
    if missing:
        raise KeyError(f"Missing training config keys: {', '.join(missing)}")

def processFolder(path, control_gatting,isControl):
    """
    Process all FCS files in a folder: load, gate, filter, and prepare training data
    Args:
        path: Directory path containing FCS files
        control_gatting: Whether to apply gating to control samples
        isControl: Whether these are control (non-plastic) samples
    Returns:
        DataFrame with processed flow cytometry measurements labeled by type
    """
    # Get list of all FCS files in directory
    files = [x for x in os.listdir(path) if x.endswith( ".fcs" )]
    data = []
    
    # Process each FCS file
    for csv in files:
        # Load FCS file with error handling
        try:
            fcs_data = flowio.FlowData(path+"/"+csv)
        except Exception as exc:
            print(f"Warning: Failed to read FCS file '{csv}': {exc}")
            continue  # Skip corrupted files and continue processing
        
        # Convert raw events to DataFrame
        npy_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
        frame = pd.DataFrame(npy_data, columns=pd.DataFrame(fcs_data.channels).iloc[0])

        # Store filename as identifier (will become plastic type label)
        file_name = os.path.basename(csv)
        frame['path'] = file_name
        
        # Apply gating based on sample type (control vs plastic standard)
        if(isControl== True):
            # Gate Controls (Non-Microplastic samples - negative class)
            if(control_gatting == True):
                print("Gating: ",os.path.basename(csv))
                before_gating = frame.shape[0]
                print("Before Gating :",before_gating)
                # Apply control-specific FSC threshold
                frame = frame[frame['FSC-A'] > model_config["training"]["fsc_limit_control"]]
                after_FSC = frame.shape[0]
                print("Gated by FSC :",before_gating-after_FSC)
                # Apply control-specific SSC threshold
                frame = frame[frame['SSC-A'] > model_config["training"]["ssc_limit_control"]]
                after_SSC = frame.shape[0]
                print("Gated by SSC :",after_FSC-after_SSC)
                print("Remaining :",after_SSC)
            data.append(frame)
        else:
            # Gate Plastic Standards (positive class with specific types)
            print("Gating: ",os.path.basename(csv))
            before_gating = frame.shape[0]
            print("Before Gating :",before_gating)
            # Apply plastic-specific FSC thresholds (min and max)
            frame = frame[frame['FSC-A'] > model_config["training"]["fsc_limit"]]
            if model_config["training"]["fsc_limit_max"] !=0:
                frame = frame[frame['FSC-A'] <= model_config["training"]["fsc_limit_max"]]
            after_FSC = frame.shape[0]
            print("Gated by FSC :",before_gating-after_FSC)
            # Apply plastic-specific SSC thresholds (min and max)
            frame = frame[frame['SSC-A'] > model_config["training"]["ssc_limit"]]
            if model_config["training"]["ssc_limit_max"] !=0:
                frame = frame[frame['SSC-A'] <= model_config["training"]["ssc_limit_max"]]
            after_SSC = frame.shape[0]
            print("Gated by SSC :",after_FSC-after_SSC)
            print("Remaining :",after_SSC)
            data.append(frame)
    
    # Combine all files into single DataFrame
    working_files = pd.concat(data, ignore_index=True,sort=False)

    # Extract type label from filename and rename column
    working_files['path'] = working_files['path'].apply(lambda x: os.path.basename(x))      
    working_files = working_files.rename(columns={'path': 'type'})

    # Exclude out-of-scale measurements that may indicate detector saturation
    print("Before top Exclusion:",working_files.shape[0])
    before_exclusion = working_files.shape[0]
    if(model_config["training"]["out_of_scale"]):
      # Check all area measurements (-A channels) for values exceeding max
      for cols in working_files.columns.tolist()[1:]:
          if(("-A"in cols) & (is_numeric_dtype(working_files[cols]))):
              working_files = working_files[(working_files[cols] < model_config["training"]["max_value"])]
    print("After top Exclusion:",working_files.shape[0])
    print("Out of Scale:", before_exclusion - working_files.shape[0])
    
    # Remove events with zero FSC-A (invalid/background events)
    print("Before FSC clean:",working_files.shape[0])
    working_files = working_files[working_files['FSC-A'] != 0]
    print("After FSC clean:",working_files.shape[0])
    
    # Exclude non-informative columns and redundant measurements
    # Keep only area measurements (-A) for training, drop scatter and other channels
    base_drop_cols = ['Time', 'SSC-H','FSC-H' ,'SSC-B-H', 'SSC-B-A', 'SSC-A','FSC-A']
    regex_drop_patterns = [
        r'Violet-H',
        r'Blue-H',
        r'YellowGreen-H',
        r'Red-H',
        r'Violet \\ Blue \\ YellowGreen \\ Red-H',
        r'Violet-A',
        r'Blue-A',
        r'YellowGreen-A',
        r'Red-A',
        r'Violet \\ Blue \\ YellowGreen \\ Red-A',
        r'-W',
        r'-H',
    ]

    # Drop base columns and collect all columns matching regex patterns
    workingDataframe = working_files.drop(columns=base_drop_cols, errors='ignore')
    regex_drop_cols = set()
    for pattern in regex_drop_patterns:
        regex_drop_cols.update(workingDataframe.filter(regex=pattern).columns)
    if regex_drop_cols:
        workingDataframe = workingDataframe.drop(columns=list(regex_drop_cols), errors='ignore')

    # Downstream compatibility
    workingDataframe.columns = workingDataframe.columns.str.replace(r"[-]", ".")
    return workingDataframe
    

def normalize_by_row(ref_data_norm):
    """
    Normalize training data per plastic type using log transformation and standard scaling
    Normalization is done separately for each type to preserve type-specific characteristics
    Args:
        ref_data_norm: DataFrame with flow cytometry measurements and 'type' column
    Returns:
        DataFrame with normalized values and type labels
    """
    allReferenceData_norm_list = []
    # Process each plastic type independently
    for plastic_type in ref_data_norm['type'].unique():

        # Get samples of this type and separate features from label
        type_data = ref_data_norm.loc[ref_data_norm['type'] == plastic_type]
        type_data = type_data.drop(columns=['type'])
        type_data_n = type_data.copy()
        
        # Apply log transformation to stabilize variance
        type_data_n = type_data_n.apply(lambda x: np.log(1+np.abs(x)))
        
        # Standardize each feature to zero mean and unit variance
        # Transpose to scale features, then transpose back
        type_data_n = StandardScaler().fit_transform(type_data_n.transpose())
        type_data_n= np.transpose(type_data_n)
        type_data_n = pd.DataFrame(type_data_n, columns=list(type_data))

        # Restore type label
        type_data_n['type'] = plastic_type
        allReferenceData_norm_list.append(type_data_n)

    # Combine all normalized types
    allReferenceData_norm = None
    allReferenceData_norm = pd.concat(allReferenceData_norm_list, ignore_index=True,sort=False)
    return allReferenceData_norm
    
    
####### Main Preprocessing Pipeline ######

def preprocess_data(standards_folder,controls_folder):
    """
    Complete training data preprocessing pipeline:
    1. Load and gate plastic standards and control samples
    2. Balance dataset by limiting samples per type
    3. Normalize data per type
    4. Remove outliers using z-score filtering
    5. Filter types with insufficient samples
    6. Balance final dataset
    Args:
        standards_folder: Path to folder with plastic standard FCS files
        controls_folder: Path to folder with control (non-plastic) FCS files
    Returns:
        DataFrame with preprocessed, balanced training data ready for modeling
    """
    # Process plastic standards (positive samples with type labels)
    workingDataframe = processFolder(standards_folder,False,False)
    print(workingDataframe.groupby('type').size())
    
    # Process control samples (negative class - non-microplastics) if provided
    if controls_folder is not None:
        workingDataframe_unknown = processFolder(controls_folder,model_config["training"]["gating_control_values"],True)
        
        # Balance control samples by limiting number per control file type
        allReferenceData_balanced_list = []
        for plastic_type in workingDataframe_unknown['type'].unique():
            type_data = workingDataframe_unknown.loc[workingDataframe_unknown['type'] == plastic_type]
            # Shuffle and limit samples per control type
            type_data = shuffle(type_data, random_state=model_config["training"]["seed"])
            type_data = type_data[:model_config["training"]["max_number_nmp_type"]]
            type_data = type_data.reset_index(drop=True)
            allReferenceData_balanced_list.append(type_data)
        workingDataframe_unknown = None
        workingDataframe_unknown = pd.concat(allReferenceData_balanced_list, ignore_index=True,sort=False)

        print(workingDataframe_unknown.groupby('type').size())
        # Label all control samples as "NMP" (Non-MicroPlastic)
        workingDataframe_unknown['type'] = "NMP"
        # Combine plastic standards with control samples
        workingDataframe = pd.concat([workingDataframe,workingDataframe_unknown],sort=False)
        print(workingDataframe.groupby('type').size())
        
    allReferenceData = None
    allReferenceData = workingDataframe

    # Normalize data per plastic type (log transform + standardization)
    allReferenceData = normalize_by_row(allReferenceData)
    
    # Statistical outlier detection and removal using z-score
    # Removes noisy measurements that deviate significantly from type distribution
    allReferenceData_outliers_list = []
    for plastic_type in allReferenceData['type'].unique():
        # Get samples of this type
        type_data = allReferenceData.loc[allReferenceData['type'] == plastic_type]
        # Skip outlier removal for NMP (controls) - preserve all negative samples
        if(plastic_type =='NMP'):
            allReferenceData_outliers_list.append(type_data)
            continue
        type_data = type_data.drop(columns=['type'])
        
        # Remove outliers beyond z-score threshold on any feature
        print("Plastic Type:",plastic_type)
        print("Before:",len(type_data.index))
        type_data_o = type_data[(np.abs(stats.zscore(type_data)) < model_config["training"]["outlier_zscore"]).all(axis=1)]
        print("After:",len(type_data_o.index))
        print("Outliers:",len(type_data.index)-len(type_data_o.index))
        
        # Restore type label
        type_data_o['type'] = plastic_type
        allReferenceData_outliers_list.append(type_data_o)
    allReferenceData = pd.concat(allReferenceData_outliers_list, ignore_index=True,sort=False)

    # Remove plastic types with insufficient training samples
    # Types with too few samples won't generalize well
    allReferenceData_list = []
    for plastic_type in allReferenceData['type'].unique():
        allReferenceData_type = allReferenceData.loc[allReferenceData['type'] == plastic_type]
        if(len(allReferenceData_type)>model_config["training"]["min_number_standards_type"]):
            allReferenceData_list.append(allReferenceData_type)
    allReferenceData = pd.concat(allReferenceData_list, ignore_index=True,sort=False)
    balanceDataset_leafs = allReferenceData.groupby('type').size()
    print(balanceDataset_leafs)
    
    # Final dataset balancing: cap samples per type to prevent class imbalance
    # Ensures no single type dominates the training set
    allReferenceData_df = []
    for plastic_type in allReferenceData['type'].unique():
        type_data = allReferenceData.loc[allReferenceData['type'] == plastic_type]
        # Limit plastic types (but keep all NMP samples)
        if(plastic_type!="NMP"):
            type_data = shuffle(type_data, random_state=model_config["training"]["seed"])
            type_data = type_data[:model_config["training"]["max_number_standards_type"]]
            type_data = type_data.reset_index(drop=True)
        allReferenceData_df.append(type_data)

    allReferenceData = pd.concat(allReferenceData_df, ignore_index=True,sort=False)

    return allReferenceData



def preprocessData(outputFile):
    """
    Main entry point for training data preprocessing
    Loads configuration, runs preprocessing pipeline, and saves results
    Args:
        outputFile: Path where preprocessed training CSV will be saved
    """
    global model_config

    # Debug output
    print([outputFile])

    # Load training configuration with gating thresholds and data limits
    with open('training_config.json') as f:
        model_config = json.load(f)

    # Validate configuration has all required parameters
    if "training" not in model_config:
        raise KeyError("model_config missing 'training' section")
    _validate_training_config(model_config["training"])

    # Set random seed for reproducible shuffling and sampling
    np.random.seed(model_config["training"]["seed"])

    # Execute full preprocessing pipeline
    allReferenceData = preprocess_data(model_config["training"]["standards_folder"],model_config["training"]["nmp_folder"])
    
    # Ensure output directory exists before saving
    output_dir = os.path.dirname(outputFile)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessed training data to CSV
    allReferenceData.to_csv(outputFile, index=False)


def main(argv):
    """
    Command-line interface for training data preprocessing
    Usage: preprocess.py -o <preprocessedFile>
    Args:
        argv: Command-line arguments
    """
    outputFile = ''
    
    # Parse command-line arguments
    try:
        opts, args = getopt.getopt(argv,"hs:n:o:d:")
    except getopt.GetoptError:
        print ('preprocess.py -o <preprocessedFile>')
        sys.exit(2)
    
    # Process each argument
    for opt, arg in opts:
        if opt == '-h':
            # Display help message
            print ('preprocess.py -o <preprocessedFile>')
            sys.exit()
        elif opt in ("-o"):
            outputFile = arg  # Output preprocessed CSV file path
    
    # Execute preprocessing
    preprocessData(outputFile)

# Entry point when script is run directly
if __name__ == "__main__":
   main(sys.argv[1:])
