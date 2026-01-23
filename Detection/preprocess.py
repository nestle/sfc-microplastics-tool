import sys, getopt, os
# Data processing libraries
import pandas as pd
import numpy as np
import json
from pandas.api.types import is_numeric_dtype
# Machine learning preprocessing
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
        "seed",              # Random seed for reproducibility
        "fsc_limit",         # Minimum forward scatter threshold
        "fsc_limit_max",     # Maximum forward scatter threshold
        "ssc_limit",         # Minimum side scatter threshold
        "ssc_limit_max",     # Maximum side scatter threshold
        "out_of_scale",      # Whether to exclude out-of-scale values
        "max_value",         # Maximum allowed value for measurements
    ]
    if not isinstance(training_cfg, dict):
        raise ValueError("model_config['training'] must be a mapping")
    missing = [key for key in required_keys if key not in training_cfg]
    if missing:
        raise KeyError(f"Missing training config keys: {', '.join(missing)}")


def processFile(file):
    """
    Process flow cytometry FCS file: load, gate, filter, and prepare data
    Args:
        file: Path to FCS file to process
    Returns:
        DataFrame with processed and filtered flow cytometry measurements
    """
    # Load FCS (Flow Cytometry Standard) file with error handling
    try:
        fcs_data = flowio.FlowData(file)
    except Exception as exc:
        raise RuntimeError(f"Failed to read FCS file '{file}': {exc}") from exc

    # Convert raw events to numpy array and create DataFrame with channel names
    npy_data = np.reshape(fcs_data.events, (-1, fcs_data.channel_count))
    frame = pd.DataFrame(npy_data, columns=pd.DataFrame(fcs_data.channels).iloc[0])

    # Gate Plastic Standards - Apply size/complexity filters based on scatter properties
    print("Gating: ",os.path.basename(file))
    before_gating = frame.shape[0]
    print("Before Gating :",before_gating)
    
    # FSC-A (Forward Scatter Area) gating - filters by particle size
    frame = frame[frame['FSC-A'] > model_config["training"]["fsc_limit"]]
    if model_config["training"]["fsc_limit_max"] !=0:
        frame = frame[frame['FSC-A'] <= model_config["training"]["fsc_limit_max"]]
    after_FSC = frame.shape[0]
    print("Gated by FSC :",before_gating-after_FSC)
    
    # SSC-A (Side Scatter Area) gating - filters by particle granularity/complexity
    frame = frame[frame['SSC-A'] > model_config["training"]["ssc_limit"]]
    if model_config["training"]["ssc_limit_max"] !=0:
        frame = frame[frame['SSC-A'] <= model_config["training"]["ssc_limit_max"]]
    after_SSC = frame.shape[0]
    print("Gated by SSC :",after_FSC-after_SSC)
    print("Remaining :",after_SSC)

    working_files = frame.copy()

    # Exclude out-of-scale measurements that may indicate saturated detector readings
    print("Before top Exclusion:",working_files.shape[0])
    before_exclusion = working_files.shape[0]
    if(model_config["training"]["out_of_scale"]):
      # Check all area measurements (-A channels) for values exceeding detector range
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
    # Keep only area measurements (-A) which are most relevant for classification
    base_drop_cols = ['Time', 'SSC-H','FSC-H' ,'SSC-B-H', 'SSC-B-A']
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

    # Replace hyphens with dots in column names for compatibility with downstream tools
    workingDataframe.columns = workingDataframe.columns.str.replace(r"[-]", ".")
    return workingDataframe
    
    

def normalize_by_row(ref_data):
    """
    Normalize flow cytometry data using log transformation and standard scaling
    Normalization is performed across features (columns) to standardize measurements
    Args:
        ref_data: DataFrame with raw flow cytometry measurements
    Returns:
        DataFrame with normalized values
    Raises:
        ValueError: If input is None or normalization fails
    """
    # Validate input data
    if ref_data is None:
        raise ValueError("Input data is None")
    if ref_data.empty:
        return ref_data.copy()

    print(list(ref_data))
    ref_data_n = ref_data.copy()
    
    # Apply log transformation to stabilize variance (log(1+|x|) handles negatives and zeros)
    ref_data_n = ref_data_n.apply(lambda x: np.log(1+np.abs(x)))
    
    # Standardize each feature (column) to zero mean and unit variance
    # Transpose to scale features, then transpose back
    try:
        ref_data_n = StandardScaler().fit_transform(ref_data_n.transpose())
    except Exception as exc:
        raise ValueError(f"Failed to normalize data: {exc}") from exc
    ref_data_n= np.transpose(ref_data_n)
    ref_data_n = pd.DataFrame(ref_data_n, columns=list(ref_data))

    return ref_data_n




def preprocessData(inputFile,outputFile,modelFolder):
    """
    Main preprocessing workflow: load config, process FCS file, normalize, and save
    Args:
        inputFile: Path to input FCS file
        outputFile: Path where preprocessed CSV will be saved
        modelFolder: Directory containing model configuration
    """
    global model_config

    # Debug output
    print([outputFile])

    # Load model configuration file containing gating and processing parameters
    with open(modelFolder+'/model_config.json') as f:
        model_config = json.load(f)

    # Validate configuration has required training parameters
    if "training" not in model_config:
        raise KeyError("model_config missing 'training' section")
    _validate_training_config(model_config["training"])

    # Set random seed for reproducibility
    np.random.seed(model_config["training"]["seed"])

    # Process FCS file: load, gate, filter
    workingDataframe = processFile(inputFile)
    
    # Normalize data using log transformation and standard scaling
    allReferenceData = normalize_by_row(workingDataframe)

    # Ensure output directory exists before saving
    output_dir = os.path.dirname(outputFile)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save preprocessed data to CSV
    allReferenceData.to_csv(outputFile, index=False)


def main(argv):
    """
    Command-line interface for FCS file preprocessing
    Usage: preprocess.py -i <inputFile> -o <outputFile> -m <modelFolder>
    Args:
        argv: Command-line arguments
    """
    outputFile = ''
    inputFile = ''
    modelFolder = ''
    
    # Parse command-line arguments
    try:
        opts, args = getopt.getopt(argv,"hs:m:o:i:")
    except getopt.GetoptError:
        print ('preprocess.py -i <inputFile> -o <outputFile> -m <modelFolder>')
        sys.exit(2)
    
    # Process each argument
    for opt, arg in opts:
        if opt == '-h':
            # Display help message
            print ('preprocess.py -i <inputFile> -o <outputFile> -m <modelFolder>')
            sys.exit()
        elif opt in ("-o"):
            outputFile = arg  # Output CSV file path
        elif opt in ("-i"):
            inputFile = arg   # Input FCS file path
        elif opt in ("-m"):
            modelFolder = arg # Model configuration directory
    
    # Execute preprocessing pipeline
    preprocessData(inputFile, outputFile,modelFolder)

# Entry point when script is run directly
if __name__ == "__main__":
   main(sys.argv[1:])
