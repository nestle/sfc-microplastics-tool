import sys, getopt
import os
# Change working directory to script location
os.chdir(sys.path[0])

# Disable GPU for testing purposes (forces CPU-only execution)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Data processing and machine learning libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import  LabelEncoder
import warnings
import keras
import json
import joblib
from scikeras.wrappers import KerasClassifier
from typing import Dict, Any
from keras.callbacks import EarlyStopping

# Initial parameters and environment configuration
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['OMP_NUM_THREADS'] = '8'  # Limit threads for OpenMP operations
warnings.filterwarnings('ignore')  # Suppress warning messages

# List of columns to exclude from model input during prediction
# These are metadata or previously computed scores not used as features
parameters_to_discard_view = ['FSC.A','SSC.A','V1.W',"nmp_score","type_score","type","plastic_color","color_score"]


# Override PU (Positive-Unlabeled) learning functions to work with the SciKeras library
# This wrapper adapts PU learning models to be compatible with scikit-learn's interface
class OverrridePUSciKeras(object):
    def __init__(self, model):
        """Initialize the PU wrapper with a base model"""
        self.model = model
        self.classes_ = [0,1]  # Binary classification classes
    
    def predict(self, X):
        """Predict class labels, converting -1 labels to 0"""
        y_pred = self.model.predict(X)
        # Convert -1 predictions (unlabeled/negative) to 0
        y_pred = np.where(y_pred==-1, 0, y_pred)
        return list(y_pred)
    
    def predict_proba(self, X):
        """Predict class probabilities in sklearn-compatible format [prob_class_0, prob_class_1]"""
        y_pred = self.model.predict_proba(X)
        pred_res = []
        for res in y_pred:
            if res > 0.5:
                # Positive class (class 1)
                if res >1:
                    pred_res.append([0,1])  # Clamp probability to 1
                else:    
                    pred_res.append([0,res])
            else:    
                # Negative class (class 0)
                pred_res.append([1-res,0])
            
        return np.array(pred_res)
    
    def fit(self, X, Y):
        """Train the underlying model"""
        self.model.fit(X,Y)
        return self
    
    def get_params(self,deep):
        """Get model parameters (sklearn interface)"""
        return {"model":self.model}
    
    def set_params(self,**params):
        """Set model parameters (sklearn interface)"""
        return self


class MLPClassifier(KerasClassifier):
    """
    Multi-Layer Perceptron Classifier using Keras backend
    Extends SciKeras KerasClassifier for sklearn compatibility
    """

    def __init__(
        self,
        hidden_layer_sizes=(1024,1024,1024,1024),  # Tuple defining neurons per hidden layer
        optimizer="adam",  # Optimization algorithm
        epochs=20,  # Maximum training epochs
        verbose=0,  # Training verbosity level
        validation_split=0.3,  # Fraction of data for validation
        callbacks = EarlyStopping,  # Callback for early stopping
        callbacks__monitor="val_loss",  # Metric to monitor for early stopping
        callbacks__patience=10,  # Epochs with no improvement before stopping
        callbacks__restore_best_weights =True,  # Restore best weights after training
        batch_size=100000,  # Training batch size
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        self.validation_split = validation_split
        self.callbacks = callbacks
        self.callbacks__monitor=callbacks__monitor
        self.callbacks__patience=callbacks__patience
        self.callbacks__restore_best_weights =callbacks__restore_best_weights
        self.batch_size = batch_size
    
    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        """Build the Keras neural network architecture"""
        model = keras.Sequential()
        # Input layer
        inp = keras.layers.Input(shape=(self.n_features_in_))
        model.add(inp)
        
        # Add hidden layers with ReLU activation
        for hidden_layer_size in self.hidden_layer_sizes:
            layer = keras.layers.Dense(hidden_layer_size, activation="relu")
            model.add(layer)
        
        # Configure output layer based on task type
        if self.target_type_ == "binary":
            n_output_units = 1
            output_activation = "sigmoid"
            loss = "binary_crossentropy"
        elif self.target_type_ == "multiclass":
            n_output_units = self.n_classes_
            output_activation = "softmax"
            loss = "categorical_crossentropy"
        else:
            raise NotImplementedError(f"Unsupported task type: {self.target_type_}")
        
        # Add output layer
        out = keras.layers.Dense(n_output_units, activation=output_activation)
        model.add(out)
        
        # Compile model with specified loss and optimizer
        model.compile(loss=loss, optimizer=compile_kwargs["optimizer"])
        return model

def load_test_model(modelfolder):
    """
    Load the trained model for plastic type classification
    Args:
        modelfolder: Path to directory containing saved models
    """
    global model_type
    global encoder_type
  
    if(os.path.exists(modelfolder+'/types.sav')):
      # Load the pre-trained type classification model
      model_type = joblib.load(modelfolder+'/types.sav')
      # Initialize label encoder with classes from config
      encoder_type = LabelEncoder()
      encoder_type.classes_ = model_config["type_classes"]
    else:
      print("Model for plastic Type classification not found..")
    
def load_test_bin_model(modelfolder):
    """
    Load the binary classification model (plastic vs non-plastic detection)
    Args:
        modelfolder: Path to directory containing saved models
    """
    global model_bin
    global encoder_bin

    if(os.path.exists(modelfolder+'/bin.sav')):
      # Load the binary classifier model
      model_bin = joblib.load(modelfolder+'/bin.sav')
      # Initialize label encoder with binary classes from config
      encoder_bin = LabelEncoder()
      encoder_bin.classes_ = model_config["bin_classes"]
    else:
        raise Warning("Model for plastic detection Not Found.")
        
def load_test_color_model(modelfolder):
    """
    Load the trained model for plastic color classification
    Args:
        modelfolder: Path to directory containing saved models
    """
    global model_color
    global encoder_color
    
    if(os.path.exists(modelfolder+'/color.sav')):
      # Load the color classification model
      model_color = joblib.load(modelfolder+'/color.sav')
      # Initialize label encoder with color classes from config
      encoder_color = LabelEncoder()
      encoder_color.classes_ = model_config["color_classes"]
    else:
      print("Model for Color classification Not found")
                
def predict_sklearn(test_data,text_labels,classifier):
    """
    Make predictions using a sklearn-compatible classifier
    Args:
        test_data: DataFrame with features to predict
        text_labels: List of class labels
        classifier: Trained classifier model
    Returns:
        DataFrame with original data plus classification and score columns
    """
    # Remove metadata/non-feature columns from input data
    x_test_water_data =test_data.copy().drop(columns=parameters_to_discard_view, errors='ignore')

    # Convert to numpy array for prediction
    x_test_water_data = np.array(x_test_water_data)
    # Get probability predictions (skip if single sample)
    if np.shape(x_test_water_data)[0] != 1:
      y_pred_score = classifier.predict_proba(x_test_water_data)
    else:
      # Default prediction for single sample edge case
      y_pred_score = np.array([[1,0]])
    
    # Process prediction probabilities into class labels and confidence scores
    prediction_rate = []
    bin_class_type = len(y_pred_score.shape)
    for res in y_pred_score:
        if (bin_class_type)>1:
          # Multi-class classifier: select class with highest probability
          prediction_rate.append([text_labels[np.argmax(res)],res[np.argmax(res)]])
        else:
          # Binary classifier: threshold at 0.5 and normalize confidence
          label = None
          if res > 0.5:
            label = text_labels[1]  # Positive class
            # Normalize score from [0.5,1] to [0,1] range
            prediction_rate.append([label,np.interp(res,[0.5,1],[0,1])])

          else:
            label = text_labels[0]  # Negative class
            # Normalize score from [0.5,1] to [0,1] range
            prediction_rate.append([label,np.interp(1-res,[0.5,1],[0,1])])

    # Create DataFrame with predictions and scores
    prediction_score_df = pd.DataFrame(prediction_rate)
    prediction_score_df.columns = ['classification','Score']

    # Combine original data with predictions
    dataset_pred = test_data
    dataset_pred = pd.concat([dataset_pred,prediction_score_df], axis=1,sort=False)
    
    return dataset_pred

    
def calculate_results(allReferenceData_norm):
    """
    Execute full classification pipeline: binary detection, type classification, and color classification
    Args:
        allReferenceData_norm: Normalized input data to classify
    Returns:
        DataFrame with classifications and confidence scores for all samples
    """
    global model_bin
    global model_type
    global model_color
    global encoder_bin
    global encoder_color
    global encoder_type

    potential_plastic = allReferenceData_norm
    nmp_data = None
    # Step 1: Binary classification (plastic vs non-plastic)
    if 'bin_classes' in model_config:
      # Predict which samples are plastic vs non-plastic
      potential_plastic = predict_sklearn(allReferenceData_norm,encoder_bin.classes_,model_bin)
      potential_plastic = potential_plastic.rename(columns={"Score": "nmp_score"})
      
      # Separate non-microplastic (NMP) samples
      nmp_data = potential_plastic.loc[potential_plastic['classification'] == encoder_bin.classes_[0]]
      nmp_data['type'] = "NMP"  # Label as non-microplastic
      nmp_data['type_score'] = 0
      nmp_data = nmp_data.drop(columns=['classification'])

      # Keep only samples classified as plastic for further classification
      potential_plastic = potential_plastic.loc[potential_plastic['classification'] == encoder_bin.classes_[1]]
      potential_plastic = potential_plastic.drop(columns=['classification','type'], errors='ignore')
      potential_plastic = potential_plastic.reset_index(drop=True)

    else:
      # Skip binary classification if not configured
      potential_plastic = allReferenceData_norm
      potential_plastic = potential_plastic.drop(columns=['type'])
  
    # Step 2: Classify plastic types for samples identified as plastic
    dataset_pred = pd.DataFrame()
    if(potential_plastic.shape[0]>0):
      # Predict specific plastic types (e.g., PE, PP, PET, etc.)
      dataset_pred = predict_sklearn(potential_plastic,encoder_type.classes_,model_type)
      dataset_pred = dataset_pred.rename(columns={"Score": "type_score","classification":"type"})

      # Merge plastic samples with non-plastic samples if both exist
      if('bin_classes' in model_config and nmp_data.shape[0]>0):
        dataset_pred = pd.concat([dataset_pred,nmp_data], ignore_index=True,sort=False)
    else:
      # No plastic samples found, return only non-plastic data
      dataset_pred = nmp_data
      
    # Step 3: Classify plastic color if color model is configured
    if 'color_classes' in model_config:
      dataset_pred = predict_sklearn(dataset_pred,model_color.classes_,model_color)
      dataset_pred = dataset_pred.rename(columns={"Score": "color_score","classification":"color"})
      # Combine type and color into single classification label (e.g., "PE_blue")
      dataset_pred['type']=dataset_pred['type'].astype(str)+'_'+dataset_pred['color'].astype(str)
      dataset_pred = dataset_pred.drop(columns=['color'])
    return dataset_pred
      
def classify_file(inputfile,outputfile,modelfolder):
  """
  Main classification workflow: load models, process input file, and save results
  Args:
      inputfile: Path to CSV file with samples to classify
      outputfile: Path where results CSV will be saved
      modelfolder: Directory containing trained models and config
  """
  # Load model configuration file
  global model_config
  with open(modelfolder+'/model_config.json') as f:
    model_config = json.load(f)
  
  # Set random seed for reproducibility
  np.random.seed(model_config["training"]["seed"])

  # Load input data from CSV
  global allReferenceData_norm
  allReferenceData_norm = pd.read_csv(inputfile)
  
  # Load models based on what's configured
  if 'type_classes' in model_config:
    # Load type classification model
    load_test_model(modelfolder)
  
  if 'bin_classes' in model_config:
    # Load binary classification model (plastic detection)
    load_test_bin_model(modelfolder)
    
  if 'color_classes' in model_config:
    # Load color classification model
    load_test_color_model(modelfolder)

  # Run classification pipeline
  result_df = calculate_results(allReferenceData_norm)
    
  # Save results to output CSV file
  result_df.to_csv(outputfile, index=False)

def main(argv):
   """
   Command-line interface for microplastic classification
   Usage: classification.py -i <inputfile> -o <outputfile> -m <modelFolder>
   Args:
       argv: Command-line arguments
   """
   inputfile = ''
   outputfile = ''
   global modelfolder
   modelfolder = ''
   
   # Parse command-line arguments
   try:
      opts, args = getopt.getopt(argv,"hi:o:m:d:",["ifile=","ofile="])
   except getopt.GetoptError:
      print ('classification.py -i <inputfile> -o <outputfile> -m <modelFolder>')
      sys.exit(2)
   
   # Process each argument
   for opt, arg in opts:
      if opt == '-h':
         # Display help message
         print ('classification.py -i <inputfile> -o <outputfile> -m <modelFolder>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg  # Input CSV file path
      elif opt in ("-o", "--ofile"):
         outputfile = arg  # Output CSV file path
      elif opt in ("-m"):
         modelfolder = arg  # Model directory path

   # Execute classification
   classify_file(inputfile,outputfile,modelfolder)

# Entry point when script is run directly
if __name__ == "__main__":
   main(sys.argv[1:])
