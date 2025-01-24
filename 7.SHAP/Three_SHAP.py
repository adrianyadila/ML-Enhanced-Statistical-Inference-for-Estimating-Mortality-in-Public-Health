import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
import keras
import pickle
import shap
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
import keras
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import layers
from keras.callbacks import Callback, ReduceLROnPlateau, EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from keras.utils import plot_model
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from keras.models import load_model
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-csv", "--datasetcsv", required=True,
	help="path to input dataset")
#ap.add_argument("-model", "--modelNNsixparameters", required=True,
	#help="path to output label binarizer")
ap.add_argument("-hist", "--hNNthreeparameters", required=True,
	help="path to output label binarizer")
args = vars(ap.parse_args())

# retrieve:    
f = open(args["hNNthreeparameters"], 'rb')
H = pickle.load(f)
f.close()

print("[INFO] loading neural network...")
modelNNthree = load_model(
    r"C:/Users/Latitude 5490/OneDrive/Documentos/NN_threeparameters_save_Adam.h5", 
    custom_objects={"r2_score": r2_score}
)

df= pd.read_csv(args["datasetcsv"])
print(df.head())

# Counting NaN values in all columns
nan_count = df.isna().sum().sum()

print('number of missing values for all variables:')
print(nan_count)

print(df.shape)

print('percentage of missing values for all variables:')
print(nan_count*100/(df.shape[0]*df.shape[1]))

#excluding variables with huge number of missing values
df2 = df.loc[:, ["var 106", "var 35", "var 34", "var 10"]] 

# Counting NaN values for new dataset
nan_count2 = df2.isna().sum().sum()

print('number of missing values for selected variables:')
print(nan_count2)

print(df2.shape)

print(df2.head())

print('percentage of missing values for selected variables:')
nan_count2*100/(df2.shape[0]*df2.shape[1])

# Finding the mean of the column having NaN
for i in df2.columns:
  mean_value = []
  if df2[i].isna().sum() > 0:
    # Replace NaNs in column S2 with the
    mean_value = df2[i].mean()

    # Replace NaNs in column S2 with the
    # mean of values in the same column
    df2[i].fillna(value=mean_value, inplace=True)

print(df2.head())

#CREATING PREDICTOR AND TARGET SETS

target = df2.iloc[:, 3].values
print('target:')
print(target)
print(target.shape)

predictor = df2.drop('var 10', axis=1)
predictor = predictor.values
print('predictor:')
print(predictor)
print(target.shape)

#NORMALIZATION

predictor_norm = StandardScaler().fit_transform(predictor)
print(predictor_norm)

# Reshape the target array to be 2D
target_2d = target.reshape(-1, 1)

# Now you can apply StandardScaler
target_norm = StandardScaler().fit_transform(target_2d)

# If you want to convert it back to a 1D array:
target_norm_1d = target_norm.flatten()
print(target_norm_1d)

#CREATING TRAINING AND TEST SETS

x_train, x_test, y_train, y_test = train_test_split(predictor_norm, target_norm_1d, test_size = 0.3, random_state = 0)

# Wrap the model
explainer = shap.Explainer(modelNNthree, x_train)

# Compute SHAP values
shap_values = explainer(x_test)

# Original array of feature names
feature_names = ["var 106", "var 35", "var 34", "var 10"]

# Mapping of variables to their corresponding descriptions
var_mapping = {
    "var 1": "pop_total",
    "var 2": "pop_urban",
    "var 3": "dens_popul",
    "var 4": "tx_urban",
    "var 5": "esp_vida_nasc",
    "var 6": "índc_envelhec",
    "var 7": "%_mort_pop_idosa",
    "var 8": "mort_até_5a",
    "var 9": "tx_mort_inf",
    "var 10": "tx_bruta_mort",
    "var 11": "tx_escol",
    "var 12": "nº_famílias_renda_até_1/2_SM",
    "var 13": "%_pobres",
    "var 14": "%_pop_pobre_CadU",
    "var 15": "%_desocupados",
    "var 16": "%_tx_emp_formal",
    "var 17": "razão_depend",
    "var 18": "gasto_pcapita_ativ_saúd",
    "var 19": "nº_estabel_saúde",
    "var 20": "nº_prof_saúde",
    "var 21": "cobert_vacinais",
    "var 22": "%_pop_atend_ESF",
    "var 23": "%_planos_saúde",
    "var 24": "%_nasc_vivos_pré_natal",
    "var 25": "%_nasc_vivos_baix_peso",
    "var 26": "%_óbitos_causas_mal_definidas",
    "var 27": "%_óbitos_causas_mal_def_s/_ass_med",
    "var 28": "nº_indústria_const_comer_serv",
    "var 29": "cobert_agrop",
    "var 30": "cobert_plant_cana_açúcar",
    "var 31": "densid_veículos",
    "var 32": "dens_reb_bovino",
    "var 33": "cobert_infraest_urba",
    "var 34": "%_cobert_vegetal",
    "var 35": "cobert_florest_plantada",
    "var 36": "%_focos_calor",
    "var 37": "%_água_trat_ETAs",
    "var 38": "%_amostras_fora_pad_microb_cons",
    "var 39": "%_amostras_fora_padr_organol_cons",
    "var 40": "IQA",
    "var 41": "cont_tóxico",
    "var 42": "%_esgoto_trat",
    "var 43": "tx_mort_cólera",
    "var 44": "tx_mort_diarréia_gastroent",
    "var 45": "tx_mort_outras_DII",
    "var 46": "[]_CO_ppb",
    "var 47": "[]_O3_ppb",
    "var 48": "[]_NO2_ppb",
    "var 49": "[]_SO2_mgm³",
    "var 50": "[]_MP2,5_mgm³",
    "var 51": "[]_MP10",
    "var 52": "temp_ºC",
    "var 53": "umidade_%",
    "var 54": "precipitação_mm",
    "var 55": "tx_mort_câncer_pulmão",
    "var 56": "nº_óbitos_gripe",
    "var 57": "nº_óbitos_pneumonia",
    "var 58": "nº_óbitos_DCVAI_bronquite_enfisema_asma",
    "var 59": "nº_óbitos_outras_DAR_sinusite_faringite_Laringite",
    "var 60": "nº_óbitos_neoplasias_traqueia_bronquios_pulmões",
    "var 61": "tx_mort_doen_resp_crian_<1a",
    "var 62": "nº_óbitos_DIC",
    "var 63": "nº_óbitos_IAM",
    "var 64": "dest_rsu_lixões",
    "var 65": "áreas_contaminadas",
    "var 66": "acidentes_amb",
    "var 67": "nº_áreas_cont_reabilit_4",
    "var 68": "obitos_intox_exogena",
    "var 69": "obitos_intox_agrotox",
    "var 70": "obitos_intox_prod_quim",
    "var 71": "nº_obtos_alimen_beb",
    "var 72": "óbitos_intox_subst_nocivas",
    "var 73": "Ídce_vulner_mudanç_climat",
    "var 74": "%_domicílios_risco_inundação",
    "var 75": "nº_óbitos_exposição_corrente_elétrica_radiac",
    "var 76": "nº_óbitos_exposição_força_natureza",
    "var 77": "índice_óbitos_eventos_hidrológicos",
    "var 78": "nº_óbitos_doenças_hipertensivas",
    "var 79": "nº_óbitos_neopl_pele",
    "var 80": "%_pop_urb_c/_abast_água",
    "var 81": "%_pop_urb_c/_esgot",
    "var 82": "%_pop_urb_c/_coleta_rdo",
    "var 83": "exist_col_seletiva",
    "var 84": "%_população_domicílios_com_banheiro_e_água_encanada",
    "var 85": "déficit_habitacional",
    "var 86": "nº_domicílios_precários",
    "var 87": "nº_domicílios_situação_coabitação_familiar",
    "var 88": "nº_domicílios_urbanos_com_pelo_menos_um_tipo_de_serviço_básico_inadequado",
    "var 89": "nº_domicílios_alugados_adensamento_excessivo",
    "var 90": "nº_domicílios_urbanos_próprios_com_adensamento_excessivo",
    "var 91": "%_pess_vuln_condiç_saneam",
    "var 92": "%_pess_s/_AEL_adeq_CadU",
    "var 93": "%_pess_s/_abast_agua_adeq_CadU",
    "var 94": "%_pess_s/_esgot_sanit_adeq_CadU",
    "var 95": "%_pess_s/_col_lixo_adeq_CadU",
    "var 96": "%_pes_dom_c/_agu_esg_inadeq",
    "var 97": "%_pes_domic_sem_energia",
    "var 98": "%_domic_c/_paredes_não_alvenaria",
    "var 99": "nº_domicílios_urbanos_sem_banheiro",
    "var 100": "%_pess_DVP_gastam_mais_1hr_trab",
    "var 101": "%_men_10a14a_filhos",
    "var 102": "%_adol_15a17a_filhos",
    "var 103": "tx_crimes_viol",
    "var 104": "tx_crimes_viol_contra_pessoa",
    "var 105": "tx_homicídios_dolosos",
    "var 106": "tx_mort_homicídio",
    "var 107": "tx_mort_agressão",
    "var 108": "tx_mort_suicídio",
    "var 109": "obtos_doenc_trab",
    "var 110": "obitos_pneumoconiose",
    "var 111": "obtos_intox_exg_exp_trab",
    "var 112": "obitos_acid_trab_grav",
    "var 113": "obtos_cancer_trab",
    "var 114": "obtos_dengue",
    "var 115": "obtos_febre_amarela",
    "var 116": "obtos_leishm",
    "var 117": "obtos_esquistoss",
    "var 118": "obtos_chagas",
    "var 119": "obtos_leptospirose",
    "var 120": "hepatites_virais",
    "var 121": "nº_óbit_alg_DIP",
    "var 122": "nº_óbit_neopl",
    "var 123": "nº_óbit_doenc_endocr_nutric_metab",
    "var 124": "nº_óbit_transt_ment_comport",
    "var 125": "nº_óbit_Dsist_nerv",
    "var 126": "nº_óbit_doenc_apar_circ",
    "var 127": "nº_óbitos_DAResp",
    "var 128": "nº_óbit_doenc_pele",
    "var 129": "nº_óbit_mal_form_congen",
    "var 130": "nº_óbit_causas_ext"
}
# Replace feature names using the mapping
mapped_feature_names = [var_mapping.get(var, var) for var in feature_names]

# Print the new mapped feature names
print(mapped_feature_names)

# Visualize
shap.summary_plot(shap_values, x_test, feature_names=mapped_feature_names, show=False)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()  # Display the plot

shap.summary_plot(shap_values, x_test, plot_type='bar', feature_names=mapped_feature_names, show=False)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()  # Display the plot

shap.dependence_plot(0, shap_values.values, x_test, show=False)
plt.tight_layout()  # Adjust layout to prevent clipping
plt.show()  # Display the plot