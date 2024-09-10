from sklearn import metrics
from itertools import combinations
import numpy as np
import operator
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import pandas as pd
EO_THRESHOLD = 0.1
DI_THRESHOLD = 0.8

TP= 'tp'
FP ='fp'
FN ='fn'
TN ='tn'


def macro_accuracy(y_true, y_predict):
    macro_acc= metrics.balanced_accuracy_score(y_true, y_predict)
    return macro_acc

def macro_f1(y_true, y_predict, labels):
    macro_f1= metrics.f1_score(y_true, y_predict,labels=labels, average='macro')
    return macro_f1

def confusion_matrix(y_true, y_predict, labels, plot=True):
    conf_matrix= metrics.confusion_matrix(y_true, y_predict, labels=labels, normalize='true')
    
    if plot:
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                                display_labels=labels)
        # disp.ax_.set_title('Normalized Confusion matrix')
        
        disp = disp.plot(cmap=plt.cm.Blues,values_format='g')
        disp.ax_.set_title('Normalized Confusion Matrix')

        plt.show()

    return conf_matrix



def compute_TP_FP_FN_TN(y_true, y_predict, labels):
    rates = {lbl:{TP: 0, FP: 0, FN: 0, TN:0} for lbl in labels}
    conf_matrix = confusion_matrix(y_true, y_predict,labels, plot=False)

    false_positives = conf_matrix.sum(axis=0) - np.diag(conf_matrix)  
    false_negatives = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    true_positives = np.diag(conf_matrix)
    true_negatives = conf_matrix.sum() - (false_positives + false_negatives + true_positives)

    # Filling dictionaries
    for i in range (len(labels)):
        rates[labels[i]][TP]= true_positives[i]
        rates[labels[i]][FP]= false_positives[i]
        rates[labels[i]][FN]= false_negatives[i]
        rates[labels[i]][TN]= true_negatives[i]
    
    return rates


def selection_rate(y_true, y_predict, labels):
    rates= compute_TP_FP_FN_TN(y_true, y_predict, labels)

    count_per_class={lbl:0 for lbl in labels}
    for item in y_true:
        count_per_class[item] += 1

    sr_per_class={lbl: rates[lbl][TP]/ count_per_class[lbl] for lbl in labels}

    return sr_per_class


def tpr(y_true, y_predict, labels):
    rates= compute_TP_FP_FN_TN(y_true, y_predict, labels)
    tpr_per_class={lbl: rates[lbl][TP]/ (rates[lbl][TP] + rates[lbl][FN]) for lbl in labels }
    
    return tpr_per_class

def fpr(y_true, y_predict, labels):
    rates= compute_TP_FP_FN_TN(y_true, y_predict, labels)
    fpr_per_class={lbl: rates[lbl][FP]/ (rates[lbl][FP] + rates[lbl][TN]) for lbl in labels}
    
    return fpr_per_class

def equalized_odds(y_true, y_predict, labels):
    print("EQUALIZED ODDS")
    tpr_values= tpr(y_true, y_predict, labels)

    fpr_values= fpr(y_true, y_predict, labels)

    equalized_odds1 = True

    eq_odds_values = {pair[0]+'--' + pair[1]: {'tpr':0, 'fpr':0} for pair in combinations(labels,2)}

    disp_matrix = np.zeros((len(labels), len(labels)))

# Crea un diccionario para almacenar los valores de igualdad de oportunidades
    eq_odds_values = {f"{label1}--{label2}": {'tpr': None, 'fpr': None} for label1 in labels for label2 in labels if label1 != label2}

    equalized_odds1 = True

# Itera sobre todas las combinaciones de clases
    for pair in combinations(labels, 2):
        first_class = pair[0]
        second_class = pair[1]
    
        if first_class == 'other' or second_class == 'other':
            continue

    # Calcular TPR
        tpr_eo_value = abs(tpr_values[first_class] - tpr_values[second_class])
        eq_odds_values[first_class + '--' + second_class]['tpr'] = tpr_eo_value 

        if tpr_eo_value >= EO_THRESHOLD:
            equalized_odds = False

        # Marca la posición en la matriz como 1 para TPR
            idx_first = labels.index(first_class)
            idx_second = labels.index(second_class)
            disp_matrix[idx_first][idx_second] = 1
            disp_matrix[idx_second][idx_first] = 1  # Simetría

    # Calcular FPR
        fpr_eo_value = abs(fpr_values[first_class] - fpr_values[second_class])
        eq_odds_values[first_class + '--' + second_class]['fpr'] = fpr_eo_value

        if fpr_eo_value >= EO_THRESHOLD:
            equalized_odds = False

        # Marca la posición en la matriz como 1 para FPR
            idx_first = labels.index(first_class)
            idx_second = labels.index(second_class)
            disp_matrix[idx_first][idx_second] = 1
            disp_matrix[idx_second][idx_first] = 1  # Simetría

    print(disp_matrix)
    
    #print('RESULTS: ', eq_odds_values)
    return disp_matrix


def disparate_impact(y_true, y_predict, labels, DI_THRESHOLD=0.8):
    # disparate impact ratio = underprivileged group SR / privileged group SR
    print("DISPARATE IMPACT.")
    disparate_impact1 = False

    sr_values = selection_rate(y_true, y_predict, labels)
    n_labels = len(labels)

    # Inicializar la matriz con ceros
    disparity_matrix = np.zeros((n_labels, n_labels))

    for pair in combinations(labels, 2):
        first_class = pair[0]
        second_class = pair[1]

        if sr_values[first_class] > sr_values[second_class]:
            pg = first_class  # Privileged group
            ug = second_class  # Underprivileged group
        else:
            pg = second_class
            ug = first_class

        disp_impact = sr_values[ug] / sr_values[pg]

        # Si hay disparidad (disparate impact), marca con 1 en la matriz
        if disp_impact < DI_THRESHOLD:
            disparate_impact1 = True
            #print(f'Disparate impact present in {ug}/{pg}\nValue: {disp_impact}')
            
            # Colocar 1 en la posición de la matriz correspondiente
            ug_idx = labels.index(ug)
            pg_idx = labels.index(pg)
            disparity_matrix[ug_idx, pg_idx] = 1

    print("Matriz de disparidad:")
    print(disparity_matrix)
    return disparity_matrix

def labelBiasMultiClass(data, labels, protectedIndex, protectedValue, targetRange):
    # Filtrar la clase protegida
    protectedClass = [(x, l) for (x, l) in zip(data, labels) if x[protectedIndex] == protectedValue]
    # Filtrar la clase no protegida
    elseClass = [(x, l) for (x, l) in zip(data, labels) if x[protectedIndex] != protectedValue]

    # Asegurarse de que ninguna clase está vacía
    if len(protectedClass) == 0 or len(elseClass) == 0:
        raise Exception("One of the classes is empty!")
    else:
        # Calcular la probabilidad de la clase targetRange en la clase protegida
        protectedProb = sum(1 for (x, l) in protectedClass if l == targetRange) / len(protectedClass)
        
        # Calcular la probabilidad de la clase targetRange en la clase no protegida
        elseProb = sum(1 for (x, l) in elseClass if l == targetRange) / len(elseClass)

    return elseProb - protectedProb

def disparate_impact_ratio(data, sensitive_feature, y_true_col, y_pred_col):
    group_protected = data[data[sensitive_feature] == 1]
    group_non_protected = data[data[sensitive_feature] == 0]

    positive_protected = np.sum(group_protected[y_pred_col] ) / len(group_protected)
    positive_non_protected = np.sum(group_non_protected[y_pred_col] ) / len(group_non_protected)
        
    dir_ratio = positive_protected / positive_non_protected 
    return dir_ratio

def chi_square_test(data, sensitive_feature, y_true_col, y_pred_col):
    # Generamos la matriz de confusión para cada grupo
    contingency_table = pd.crosstab(data[sensitive_feature], data[y_pred_col] >= 4)
    
    # Test de Chi-cuadrado
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    return chi2, p