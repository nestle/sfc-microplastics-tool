"""Training module for microplastic detection classifiers.

This module trains three classification models using stratified k-fold cross-validation:
1. Binary classifier: Plastic vs Non-Microplastic (NMP) using positive-unlabeled learning
2. Type classifier: Classifies plastic types (PE, PET, PP, etc.)
3. Color classifier: Classifies plastic colors (red, blue, green, etc.)

Models use ensemble methods (BaggingClassifier with MLPClassifier) and are evaluated
using confusion matrices and precision/recall/fscore metrics across all k-folds.
"""

import sys, getopt, os
os.chdir(sys.path[0])  # Change to script directory for relative path resolution

# Configure TensorFlow to use CPU only (disable GPU for consistent performance)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['OMP_NUM_THREADS'] = '8'  # Limit OpenMP threads for CPU parallelism

# Machine learning and neural network imports
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
import json
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import shuffle
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from pulearn import ElkanotoPuClassifier
import joblib
from scikeras.wrappers import KerasClassifier
from sklearn.ensemble import BaggingClassifier
from typing import Dict, Any
import tensorflow
import scikeras
from datetime import datetime

# Print library versions for reproducibility tracking
print("Tensorflow Version : {}".format(tensorflow.__version__))
print("Scikeras Version : {}".format(scikeras.__version__))

# Set numpy to print full arrays (useful for debugging large outputs)
np.set_printoptions(threshold=sys.maxsize)

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
        "seed",
        "min_number_standards_type",
        "train_type",
        "number_ensemble_type",
        "number_ensemble_color",
        "number_ensemble_binary",
        "train_binary",
        "train_color",
    ]
    if not isinstance(training_cfg, dict):
        raise ValueError("model_config['training'] must be a mapping")
    missing = [key for key in required_keys if key not in training_cfg]
    if missing:
        raise KeyError(f"Missing training config keys: {', '.join(missing)}")

# Override PU (Positive-Unlabeled) learning functions to work with the SciKeras library
# This wrapper adapts ElkanotoPuClassifier for compatibility with BaggingClassifier
class OverrridePUSciKeras(object):
    """Wrapper for PU learning classifier to work with scikit-learn ensemble methods.
    
    Converts the Elkanoto PU classifier interface to be compatible with BaggingClassifier
    by implementing required scikit-learn estimator methods and converting predictions.
    """
    
    def __init__(self, model):
        """Initialize wrapper with the PU learning model.
        
        Args:
            model: ElkanotoPuClassifier instance to wrap
        """
        self.model = model
        self.classes_ = [0, 1]  # Binary classification: 0=NMP (unlabeled), 1=Plastic (positive)
    
    def predict(self, X):
        """Make binary predictions on new data.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Predictions where -1 is converted to 0 (NMP) and 1 stays as 1 (Plastic)
        """
        y_pred = self.model.predict(X)
        y_pred = np.where(y_pred == -1, 0, y_pred)  # Convert -1 (negative) to 0 (NMP)
        return list(y_pred)
    def predict_proba(self, X):
        """Get probability estimates for both classes.
        
        Args:
            X: Input features (n_samples, n_features)
            
        Returns:
            Probability matrix where each row is [prob_class_0, prob_class_1]
        """
        y_pred = self.model.predict_proba(X)
        pred_res = []
        for res in y_pred:
            if res > 0.5:
                # Confidence toward class 1 (Plastic)
                if res > 1:
                    pred_res.append([0, 1])
                else:    
                    pred_res.append([0, res])
            else:
                # Confidence toward class 0 (NMP)
                pred_res.append([1 - res, 0])
            
        return np.array(pred_res)
    def fit(self, X, Y):
        """Train the PU learning model.
        
        Args:
            X: Input features (n_samples, n_features)
            Y: Labels where 1=positive (Plastic), 0/unlabeled (NMP)
            
        Returns:
            self (for sklearn estimator interface)
        """
        self.model.fit(X, Y)
        return self
    
    def get_params(self, deep):
        """Get model parameters for scikit-learn compatibility."""
        return {"model": self.model}
    
    def set_params(self, **params):
        """Set model parameters for scikit-learn compatibility."""
        return self


class MLPClassifier(KerasClassifier):
    """Multi-layer perceptron classifier using Keras/TensorFlow.
    
    Builds sequential neural networks with configurable hidden layers, regularization,
    and early stopping. Handles both binary and multiclass classification tasks.
    """

    def __init__(
        self,
        hidden_layer_sizes=(64, 64),
        optimizer=tf.keras.optimizers.Adam(),
        epochs=10,
        verbose=1,
        validation_split=0.1,
        callbacks=EarlyStopping,
        callbacks__monitor="val_loss",
        callbacks__patience=20,
        callbacks__restore_best_weights=True,
        batch_size=5000,
        shuffle=True,
        **kwargs,
    ):
        """Initialize MLP classifier with network architecture and training parameters.
        
        Args:
            hidden_layer_sizes: Tuple of units per hidden layer (default: 2 layers, 64 units each)
            optimizer: TensorFlow optimizer (default: Adam)
            epochs: Number of training epochs (default: 10)
            verbose: Verbosity level for training output (default: 1)
            validation_split: Fraction of data for validation (default: 0.1)
            callbacks: Keras callback classes to use (default: EarlyStopping)
            callbacks__monitor: Metric to monitor for early stopping (default: val_loss)
            callbacks__patience: Epochs to wait before early stopping (default: 20)
            callbacks__restore_best_weights: Restore weights from best epoch (default: True)
            batch_size: Training batch size (default: 5000)
            shuffle: Whether to shuffle training data (default: True)
        """
        super().__init__(**kwargs)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        self.validation_split = validation_split
        self.callbacks = callbacks
        self.callbacks__monitor = callbacks__monitor
        self.callbacks__patience = callbacks__patience
        self.callbacks__restore_best_weights = callbacks__restore_best_weights
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def _keras_build_fn(self, compile_kwargs: Dict[str, Any]):
        """Build and compile the Keras neural network model.
        
        Creates sequential model with:
        - Input layer matching number of flow cytometry features
        - Hidden layers with ReLU activation and L2 regularization (0.0001)
        - Output layer appropriate for task type (binary or multiclass)
        
        Args:
            compile_kwargs: Additional compilation keyword arguments
            
        Returns:
            Compiled Keras Sequential model
        """
        model = keras.Sequential()
        # Input layer with feature dimension
        inp = keras.layers.Input(shape=(self.n_features_in_,))
        model.add(inp)
        
        # Add hidden layers with ReLU activation and L2 regularization
        for hidden_layer_size in self.hidden_layer_sizes:
            layer = keras.layers.Dense(
                hidden_layer_size,
                activation="relu",
                kernel_regularizer=l2(0.0001)  # L2 penalty to prevent overfitting
            )
            model.add(layer)
        
        # Configure output layer based on task type
        if self.target_type_ == "binary":
            # Binary classification: single output with sigmoid
            n_output_units = 1
            output_activation = "sigmoid"
            loss = "binary_crossentropy"
        elif self.target_type_ == "multiclass":
            # Multiclass: output per class with softmax
            print(self.n_classes_)
            n_output_units = self.n_classes_
            output_activation = "softmax"
            loss = "sparse_categorical_crossentropy"
        else:
            raise NotImplementedError(f"Unsupported task type: {self.target_type_}")
        
        # Add output layer
        out = keras.layers.Dense(n_output_units, activation=output_activation)
        model.add(out)
        
        # Compile model with specified loss function and optimizer
        model.compile(loss=loss, optimizer=self.optimizer)
        return model

def get_x_random_samples(input_data, numSamples, seed):
    """Balance dataset by randomly sampling equal number of samples per plastic type.
    
    This helper function ensures equal representation of all plastic types by randomly
    selecting numSamples from each type, useful for addressing class imbalance.
    
    Args:
        input_data: DataFrame with 'type' column containing plastic type labels
        numSamples: Number of samples to select per type
        seed: Random seed for reproducible shuffling
        
    Returns:
        Balanced DataFrame with equal samples per plastic type
    """
    balanced_list = []
    # Sample from each plastic type independently
    for plastic_type in input_data['type'].unique():
        type_data = input_data.loc[input_data['type'] == plastic_type]
        type_data = shuffle(type_data, random_state=seed)  # Shuffle for random selection
        type_data = type_data[:numSamples]  # Take first numSamples after shuffle
        type_data = type_data.reset_index(drop=True)
        balanced_list.append(type_data)
    input_data = None
    # Combine all balanced types
    input_data = pd.concat(balanced_list, ignore_index=True, sort=False)
    return input_data
  


##################################################
############### Model Training Type ##############
##################################################
def trainModelType(allReferenceData, outputFolder, seed):
  """Train plastic type classifier using ensemble method with k-fold cross-validation.
  
  Trains a BaggingClassifier with MLP base estimators to classify plastic types
  (PE, PET, PP, etc.). Excludes NMP samples and applies stratified k-fold validation.
  Saves model and computes precision/recall/fscore metrics.
  
  Args:
      allReferenceData: DataFrame with flow cytometry features and 'type' column
      outputFolder: Directory to save trained model
      seed: Random seed for reproducible results
      
  Returns:
      DataFrame of training data after filtering
  """
  # Create output directory if it doesn't exist
  os.makedirs(outputFolder, exist_ok=True)
  
  # Load minimum sample threshold from config
  min_samples_by_type = model_config["training"]["min_number_standards_type"]
  
  # Exclude NMP samples (non-microplastic) - only train on plastic types
  allReferenceData = allReferenceData.loc[allReferenceData['type'] != "NMP"]
  
  # Print initial class distribution
  print(allReferenceData.groupby('type').size())
  
  # If train_type enabled, extract main plastic type from filename (e.g., PE_red.fcs -> PE)
  if model_config["training"]["train_type"]:
    allReferenceData["type"] = allReferenceData["type"].str.split('_').str[0]  # Split on underscore
    allReferenceData["type"] = allReferenceData["type"].str.split('.').str[0]  # Split on dot
    print(allReferenceData.groupby('type').size())
  
  # Filter out plastic types with insufficient samples for reliable training
  allReferenceData_list = []
  for plastic_type in allReferenceData['type'].unique():
      allReferenceData_type = allReferenceData.loc[allReferenceData['type'] == plastic_type]
      if len(allReferenceData_type) >= min_samples_by_type:
          allReferenceData_list.append(allReferenceData_type)
  allReferenceData = pd.concat(allReferenceData_list, ignore_index=True, sort=False)
  print(allReferenceData.groupby('type').size())

  # Store training data for reference
  type_train_df = allReferenceData.copy()
  
  # Clear TensorFlow session to avoid memory issues
  K.clear_session()
    
  # Shuffle data for randomization
  allReferenceData = shuffle(allReferenceData, random_state=seed)
  
  # Prepare features (X) and labels (y) for training
  x = np.array(allReferenceData.drop(columns=['type']))  # All columns except type
  y = np.array(allReferenceData['type'])  # Plastic type labels
  
  # Configure k-fold cross-validation for robust model evaluation
  n_folds = 3
  
  # Instantiate the cross validator
  kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
  # Accumulate predictions from all folds for robust evaluation
  y_test_list = []
  y_pred_list = []
  
  number_ensemble = model_config["training"]["number_ensemble_type"]
  
  for fold, (train, test) in enumerate(kfold.split(x, y)):  
      print(f"Training Type Model - Fold {fold + 1}/{n_folds}")
      # Generate batches from indices
      x_train, x_test = x[train], x[test]
      y_train, y_test = y[train], y[test]
      
      # Clear session before each fold
      K.clear_session()
      
      # Train model on this fold
      model = BaggingClassifier(base_estimator=MLPClassifier(), n_estimators=number_ensemble, random_state=seed, n_jobs=1)
      model = model.fit(np.array(x_train),np.array(y_train))
      y_pred = model.predict(x_test)
      
      # Accumulate results from this fold
      y_test_list.append(y_test)
      y_pred_list.append(y_pred)
  
  # Concatenate results from all folds for final evaluation
  y_test_all = np.concatenate(y_test_list)
  y_pred_all = np.concatenate(y_pred_list)
  
  # Save model trained on first fold (or retrain on full data)
  model = BaggingClassifier(base_estimator=MLPClassifier(), n_estimators=number_ensemble, random_state=seed, n_jobs=1)
  model = model.fit(np.array(x),np.array(y))
  joblib.dump(model, outputFolder+"/types.sav")

  global type_classes
  type_classes = model.classes_.tolist()

  cnf_matrix = confusion_matrix(y_test_all, y_pred_all)
  cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

  #Stats
  print("#### Stats ####")
  
  print("#Confusion Matrix")
  print(cnf_matrix)
  print("#Confusion Matrix Normalized")
  print(cnf_matrix_norm)
  
  precision, recall, fscore, support = score(y_test_all, y_pred_all)

  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(fscore))
  print('support: {}'.format(support))

  print('Macro precision:',np.average(precision))
  print('Macro recall:',np.average(recall))
  print('Macro fscore:',np.average(fscore))
  
  reference_data_df = pd.DataFrame(allReferenceData.groupby('type').size())
  reference_data_df = reference_data_df.reset_index()
  global type_stats
  type_stats ={
    "training_input": np.asarray(reference_data_df).tolist(),
    "confusion_matrix": cnf_matrix.tolist(),
    "confusion_matrix_norm": cnf_matrix_norm.tolist(),
    "precision": np.average(precision),
    "recall": np.average(recall),
    "fscore": np.average(fscore)
  }
  return type_train_df



##################################################
############### Model Training Color #############
##################################################
def trainModelColor(allReferenceData, outputFolder, seed):
  """Train plastic color classifier using ensemble method with k-fold cross-validation.
  
  Trains a BaggingClassifier with MLP base estimators to classify plastic colors
  (red, blue, green, etc.). Excludes NMP samples and applies stratified k-fold validation.
  Saves model and computes precision/recall/fscore metrics.
  
  Args:
      allReferenceData: DataFrame with flow cytometry features and 'type' column
      outputFolder: Directory to save trained model
      seed: Random seed for reproducible results
      
  Returns:
      DataFrame of training data after filtering
  """
  # Create output directory if it doesn't exist
  os.makedirs(outputFolder, exist_ok=True)
  
  # Minimum number of samples required per color class for reliable training
  min_samples_by_color = 100
  
  # Exclude NMP samples (non-microplastic) - only train on plastic types
  allReferenceData = allReferenceData.loc[allReferenceData['type'] != "NMP"]
  
  # Print initial class distribution
  print(allReferenceData.groupby('type').size())
  
  # Extract color from filename (e.g., PE_red.fcs -> red)
  allReferenceData["type"] = allReferenceData["type"].str.split('_').str[1]  # Get color after underscore
  allReferenceData["type"] = allReferenceData["type"].str.split('.').str[0]  # Remove file extension
  print(allReferenceData.groupby('type').size())

  # Filter out color classes with insufficient samples for reliable training
  allReferenceData_list = []
  for plastic_color in allReferenceData['type'].unique():
      allReferenceData_color = allReferenceData.loc[allReferenceData['type'] == plastic_color]
      if len(allReferenceData_color) >= min_samples_by_color:
          allReferenceData_list.append(allReferenceData_color)
  allReferenceData = pd.concat(allReferenceData_list, ignore_index=True, sort=False)
  print(allReferenceData.groupby('type').size())
  
  # Store training data for reference
  color_train_df = allReferenceData.copy()
  
  # Clear TensorFlow session to avoid memory issues
  K.clear_session()
  
  # Shuffle data for randomization
  allReferenceData = shuffle(allReferenceData, random_state=seed)

  # Prepare features (X) and labels (y) for training
  x = np.array(allReferenceData.drop(columns=['type']))  # All columns except type
  y = np.array(allReferenceData['type'])  # Color labels
    
  # Configure k-fold cross-validation for robust model evaluation
  n_folds = 3

  # Instantiate the cross validator
  kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
  # Accumulate predictions from all folds for robust evaluation
  y_test_list = []
  y_pred_list = []

  number_ensemble = model_config["training"]["number_ensemble_color"]

  for fold, (train, test) in enumerate(kfold.split(x, y)):
      print(f"Training Color Model - Fold {fold + 1}/{n_folds}")
      # Generate batches from indices
      x_train, x_test = x[train], x[test]
      y_train, y_test = y[train], y[test]
      
      # Clear session before each fold
      K.clear_session()
      
      # Train model on this fold
      model = BaggingClassifier(base_estimator=MLPClassifier(), n_estimators=number_ensemble, random_state=seed, n_jobs=1)
      model = model.fit(np.array(x_train),np.array(y_train))
      y_pred = model.predict(x_test)
      
      # Accumulate results from this fold
      y_test_list.append(y_test)
      y_pred_list.append(y_pred)
  
  # Concatenate results from all folds for final evaluation
  y_test_all = np.concatenate(y_test_list)
  y_pred_all = np.concatenate(y_pred_list)
  
  # Save model trained on full data
  model = BaggingClassifier(base_estimator=MLPClassifier(), n_estimators=number_ensemble, random_state=seed, n_jobs=1)
  model = model.fit(np.array(x),np.array(y))
  joblib.dump(model, outputFolder+"/color.sav")
  

  global color_classes
  color_classes = model.classes_.tolist()

  cnf_matrix = confusion_matrix(y_test_all, y_pred_all)
  cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

  #Stats
  print("#### Color Stats ####")
  
  print("#Confusion Matrix")
  print(cnf_matrix)
  print("#Confusion Matrix Normalized")
  print(cnf_matrix_norm)
  
  precision, recall, fscore, support = score(y_test_all, y_pred_all)

  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(fscore))
  print('support: {}'.format(support))

  print('Macro precision:',np.average(precision))
  print('Macro recall:',np.average(recall))
  print('Macro fscore:',np.average(fscore))
  
  reference_data_df = pd.DataFrame(allReferenceData.groupby('type').size())
  reference_data_df = reference_data_df.reset_index()
  global color_stats
  color_stats ={
    "training_input": np.asarray(reference_data_df).tolist(),
    "confusion_matrix": cnf_matrix.tolist(),
    "confusion_matrix_norm": cnf_matrix_norm.tolist(),
    "precision": np.average(precision),
    "recall": np.average(recall),
    "fscore": np.average(fscore)
  }
  return color_train_df

####################################################
############### Model Training Binary ##############
####################################################

def trainModelBin(allReferenceData, outputFolder, seed):
  """Train binary plastic classifier using PU learning with k-fold cross-validation.
  
  Trains a binary classifier to distinguish plastics from non-microplastics (NMP).
  Uses positive-unlabeled (PU) learning via ElkanotoPuClassifier wrapped for compatibility.
  Applies stratified k-fold validation and balances class distribution if needed.
  
  Args:
      allReferenceData: DataFrame with flow cytometry features and 'type' column
      outputFolder: Directory to save trained model
      seed: Random seed for reproducible results
      
  Returns:
      DataFrame of training data after label conversion to binary (0/1)
  """
  # Create output directory if it doesn't exist
  os.makedirs(outputFolder, exist_ok=True)
  
  # Balance dataset if plastic count exceeds NMP count
  # This addresses class imbalance in positive-unlabeled learning scenarios
  number_plastics = allReferenceData[allReferenceData.type != 'NMP'].shape[0]
  number_unknown = allReferenceData[allReferenceData.type == 'NMP'].shape[0]
  
  if number_plastics > number_unknown:
    # Downsample plastics to match NMP count equally across types
    number_standards = allReferenceData['type'].nunique() - 1  # Exclude NMP
    number_per_standard = int(number_unknown / number_standards)  # Samples per plastic type
  
    allReferenceData_df = []
    for plastic_type in allReferenceData['type'].unique():
        type_data = allReferenceData.loc[allReferenceData['type'] == plastic_type]
        if(plastic_type!="NMP"):
            type_data = shuffle(type_data, random_state=seed)
            type_data = type_data[:number_per_standard]
            type_data = type_data.reset_index(drop=True)
        allReferenceData_df.append(type_data)
        
    allReferenceData = pd.concat(allReferenceData_df, ignore_index=True,sort=False)
  
  # Print class distribution before binary conversion
  print(allReferenceData.groupby('type').size())
  
  # Convert to binary labels: 0=NMP (unlabeled/negative), 1=Plastic (positive)
  allReferenceData['type'] = allReferenceData['type'].str.replace(r'^(?!NMP).*$', '1', regex=True)  # Non-NMP -> 1
  allReferenceData['type'] = allReferenceData['type'].str.replace('NMP', '0', regex=True)  # NMP -> 0
  allReferenceData['type'] = allReferenceData['type'].astype(int)
  print(allReferenceData.groupby('type').size())
  
  # Store training data for reference
  bin_train_df = allReferenceData.copy()
  
  # Shuffle data for randomization
  allReferenceData = shuffle(allReferenceData, random_state=seed)

  # Prepare features (X) and labels (y) for training
  x = np.array(allReferenceData.drop(columns=['type']))
  y = np.array(allReferenceData['type'])
  
  #K-fold Cross Validation
  n_folds=3
  y_test_list = []
  y_pred_list = []
  
  # Instantiate the cross validator
  kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

  number_ensemble = model_config["training"]["number_ensemble_binary"]
  
  for fold, (train, test) in enumerate(kfold.split(x, y)):
      print(f"Training Binary Model - Fold {fold + 1}/{n_folds}")
      # Generate batches from indices
      x_train, x_test = x[train], x[test]
      y_train, y_test = y[train], y[test]
      
      # Clear session before each fold
      K.clear_session()
      
      print("Train/test stats:",x_train.shape, x_test.shape, y_train.shape, y_test.shape)
      clf = OverrridePUSciKeras(ElkanotoPuClassifier(MLPClassifier()))
      model = BaggingClassifier(base_estimator=clf, n_estimators=number_ensemble, random_state=seed, n_jobs=1)
      model = model.fit(np.array(x_train),np.array(y_train))
      y_pred = model.predict(x_test)
      
      # Accumulate results from this fold
      y_test_list.append(y_test)
      y_pred_list.append(y_pred)
  
  # Concatenate results from all folds for final evaluation
  y_test_all = np.concatenate(y_test_list)
  y_pred_all = np.concatenate(y_pred_list)
  
  # Save model trained on full data
  clf = OverrridePUSciKeras(ElkanotoPuClassifier(MLPClassifier()))
  model = BaggingClassifier(base_estimator=clf, n_estimators=number_ensemble, random_state=seed, n_jobs=1)
  model = model.fit(np.array(x),np.array(y))
  joblib.dump(model, outputFolder+"/bin.sav")

  global bin_classes
  bin_classes = ["NMP","Plastic"]

  #Stats
  cnf_matrix = confusion_matrix(y_test_all, y_pred_all)
  cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]
  print("#### Binary Stats ####")

  print("#Confusion Matrix")
  print(cnf_matrix)
  print("#Confusion Matrix Normalized")
  print(cnf_matrix_norm)
  precision, recall, fscore, support = score(y_test_all, y_pred_all)

  print('precision: {}'.format(precision))
  print('recall: {}'.format(recall))
  print('fscore: {}'.format(fscore))
  print('support: {}'.format(support))

  print('Macro precision:',np.average(precision))
  print('Macro recall:',np.average(recall))
  print('Macro fscore:',np.average(fscore))

  reference_data_df = pd.DataFrame(allReferenceData.groupby('type').size())
  reference_data_df = reference_data_df.reset_index()
  global bin_stats
  bin_stats ={
  "training_input": np.asarray(reference_data_df).tolist(),
  "confusion_matrix": cnf_matrix.tolist(),
  "confusion_matrix_norm": cnf_matrix_norm.tolist(),
  "precision": np.average(precision),
  "recall": np.average(recall),
  "fscore": np.average(fscore)
  }

  y_test_list.append(y_test)
  y_pred_list.append(y_pred)
      

  return bin_train_df

#############################################
############### Model Training ##############
#############################################

def trainModel(preprocessedFile, outputFolder):
  """Main training function orchestrating all classifier training pipelines.
  
  Loads configuration, trains three classifiers (binary, type, color) using
  k-fold cross-validation, and saves models along with statistics and configuration.
  
  Args:
      preprocessedFile: Path to preprocessed CSV file with flow cytometry data
      outputFolder: Directory to save trained models, config, and statistics
      
  Raises:
      KeyError: If training_config.json missing 'training' section or required keys
  """
  # Initialize global variables for storing model information across functions
  global type_classes
  type_classes = None
  global bin_classes
  bin_classes = None
  global color_classes
  color_classes = None
  global type_stats
  type_stats = None
  global bin_stats
  bin_stats = None
  global color_stats
  color_stats = None
  global model_config
  
  # Print training start time for progress tracking
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  print("Start Training :", current_time)
  
  # Create output directory if it doesn't exist
  os.makedirs(outputFolder, exist_ok=True)
  
  # Load training configuration from JSON file
  with open("./training_config.json") as data_file:
    model_config = json.load(data_file)
  
  # Validate configuration structure
  if "training" not in model_config:
    raise KeyError("model_config missing 'training' section")
  _validate_training_config(model_config["training"])
  
  # Set random seeds for reproducibility
  np.random.seed(model_config["training"]["seed"])
  seed = model_config["training"]["seed"]
  
  # Load preprocessed training data
  allReferenceData = pd.read_csv(preprocessedFile)
  
  # Train binary classifier (Plastic vs NMP) if enabled
  print("########### Training Binary Classifier ########")
  if model_config["training"]["train_binary"]:
    bin_train_df = trainModelBin(allReferenceData.copy(), outputFolder, seed)
    model_config["bin_classes"] = bin_classes

  # Train type classifier (plastic material types) - always enabled
  print("########### Training Type Classifier ########")
  type_train_df = trainModelType(allReferenceData.copy(), outputFolder, seed)
  model_config["type_classes"] = type_classes

  # Train color classifier if enabled
  print("########### Training Color Classifier ########")
  if model_config["training"]["train_color"]:
    color_train_df = trainModelColor(allReferenceData.copy(), outputFolder, seed)
    model_config["color_classes"] = color_classes

  # Print training end time
  now = datetime.now()
  current_time = now.strftime("%H:%M:%S")
  print("Finish Training :", current_time)

  # Save updated model configuration with trained class labels
  with open(outputFolder + "/model_config.json", "w") as data_file:
    json.dump(model_config, data_file)
  
  # Aggregate and save training statistics from all classifiers
  training_stats = {
    "bin_stats": bin_stats,
    "type_stats": type_stats,
    "color_stats": color_stats
  }
  with open(outputFolder + "/stats.json", "w") as stats_file:
    json.dump(training_stats, stats_file)

def main(argv):
  """Parse command-line arguments and launch training.
  
  Command-line usage:
      python trainModels.py -i <preprocessedFile> -o <outputFolder>
  
  Args:
      argv: Command-line arguments (sys.argv[1:])
      
  Options:
      -h: Show help message
      -i: Path to preprocessed training data CSV file
      -o: Output directory for trained models and statistics
  """
  preprocessedFile = ''
  outputFolder = ''
  
  try:
    opts, args = getopt.getopt(argv, "hi:o:m:d:")
  except getopt.GetoptError:
    print('trainModel.py -i <preprocessedFile>  -o <outputFolder> ')
    sys.exit(2)
  
  # Parse command-line options
  for opt, arg in opts:
    if opt == '-h':
      print('trainModel.py -i <preprocessedFile> -o <outputFolder> ')
      sys.exit()
    elif opt in ("-i"):
      preprocessedFile = arg
    elif opt in ("-o"):
      outputFolder = arg

  # Launch training with parsed arguments
  trainModel(preprocessedFile, outputFolder)


if __name__ == "__main__":
  main(sys.argv[1:])
