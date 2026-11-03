import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d

class BidirectionalMilkModel:
    def __init__(self, csv_path=None):
        if csv_path is None:
            # Check common file names
            import os
            if os.path.exists('milk_sensor_dataset.csv'):
                csv_path = 'milk_sensor_dataset.csv'
            elif os.path.exists('Dataset.csv'):
                csv_path = 'Dataset.csv'
            else:
                csv_path = 'milk_sensor_dataset.csv'
        self.csv_path = csv_path
        self.load_and_prep_data()
        self.fit_models()

    def load_and_prep_data(self):
        df = pd.read_csv(self.csv_path)
        # Compute Bandwidth BW (GHz) = f1 / Q1
        df['BW_GHz'] = df['f1_GHz'] / df['Q1']
        df['BW_MHz'] = df['BW_GHz'] * 1000.0
        self.df = df
        
        # Filter training data for classification & regression
        # We classify adulterants: Water, Sugar, MixedElectrolyte, Milk (Fresh), SyntheticMilk
        self.classes = sorted(self.df['material_type'].unique())
        
    def fit_models(self):
        # 1. Forward Classifier: (f1, depth1, Q1, BW) -> material_type
        features = ['f1_GHz', 'depth1_dB', 'Q1', 'BW_MHz']
        X = self.df[features].values
        y_class = self.df['material_type'].values
        
        self.clf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.clf.fit(X, y_class)
        
        # 2. Forward Concentration Regressors (separate per material_type for maximum precision)
        # We also fit a global regressor as fallback
        self.conc_models = {}
        for mat in self.classes:
            sub = self.df[self.df['material_type'] == mat]
            if len(sub) >= 2 and sub['concentration'].nunique() > 1:
                # Use interpolation spline or polynomial for smooth arbitrary concentration prediction
                sub_sorted = sub.sort_values(by='f1_GHz')
                # Fit multi-feature regression or 1D monotonic spline based on resonant shift
                # In microwave sensors, resonant frequency f1 or depth shift is strongly monotonic with concentration
                try:
                    # Interpolation & Extrapolation function
                    f_interp = interp1d(sub_sorted['f1_GHz'], sub_sorted['concentration'], 
                                        kind='linear', fill_value='extrapolate')
                    self.conc_models[mat] = f_interp
                except Exception:
                    pass
        
        # 3. Inverse Models: (material_type, concentration) -> (f1_GHz, depth1_dB, Q1, BW_MHz, eps_r_real)
        # For each material, fit continuous interpolator over concentration
        self.inv_models = {}
        target_cols = ['f1_GHz', 'depth1_dB', 'Q1', 'eps_r_real']
        for mat in self.classes:
            sub = self.df[self.df['material_type'] == mat].sort_values(by='concentration')
            mat_dict = {}
            if len(sub) == 1:
                # Constant response
                row = sub.iloc[0]
                for col in target_cols:
                    mat_dict[col] = lambda c, val=row[col]: np.full_like(np.array(c, dtype=float), val)
            else:
                for col in target_cols:
                    mat_dict[col] = interp1d(sub['concentration'], sub[col], kind='linear', fill_value='extrapolate')
            self.inv_models[mat] = mat_dict

    def predict_forward(self, f1_GHz, depth1_dB, Q1=None, BW_MHz=None):
        """Given sensor microwave parameters, predict adulterant and concentration."""
        if Q1 is None and BW_MHz is not None:
            Q1 = (f1_GHz * 1000.0) / BW_MHz
        elif BW_MHz is None and Q1 is not None:
            BW_MHz = (f1_GHz / Q1) * 1000.0
        elif Q1 is None and BW_MHz is None:
            Q1 = 10.0
            BW_MHz = (f1_GHz / Q1) * 1000.0

        features = np.array([[f1_GHz, depth1_dB, Q1, BW_MHz]])
        pred_class = self.clf.predict(features)[0]
        probs = dict(zip(self.clf.classes_, self.clf.predict_proba(features)[0]))
        
        # Predict concentration
        pred_conc = 0.0
        if pred_class in self.conc_models:
            pred_conc = float(self.conc_models[pred_class](f1_GHz))
            pred_conc = max(0.0, pred_conc)
        elif pred_class in ['Milk', 'FreshMilk', 'SyntheticMilk', 'Baseline']:
            pred_conc = 0.0
            
        return {
            'predicted_adulterant': pred_class,
            'predicted_concentration': round(pred_conc, 2),
            'class_probabilities': probs,
            'computed_Q1': round(Q1, 2),
            'computed_BW_MHz': round(BW_MHz, 2)
        }

    def predict_inverse(self, adulterant, concentration):
        """Given adulterant and concentration (even values never in dataset), predict sensor response."""
        if adulterant not in self.inv_models:
            raise ValueError(f"Unknown adulterant: {adulterant}. Available: {self.classes}")
            
        model_dict = self.inv_models[adulterant]
        f1 = float(model_dict['f1_GHz'](concentration))
        depth = float(model_dict['depth1_dB'](concentration))
        q1 = float(model_dict['Q1'](concentration))
        eps_r = float(model_dict['eps_r_real'](concentration))
        bw_mhz = (f1 / q1) * 1000.0
        
        return {
            'adulterant': adulterant,
            'concentration': concentration,
            'predicted_f1_GHz': round(f1, 4),
            'predicted_depth1_dB': round(depth, 3),
            'predicted_Q1': round(q1, 2),
            'predicted_BW_MHz': round(bw_mhz, 2),
            'predicted_eps_r_real': round(eps_r, 2)
        }
