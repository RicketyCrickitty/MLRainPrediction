# IMPORTS & MACRO OPTIONS
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from mlxtend.frequent_patterns import apriori
import tabulate

pd.set_option("display.max_columns", None)
RANDOM_SEED = 5805
### CHANGE THIS FOR GRADING ###
DATA_PATH = 'weatherAUS.csv'

# %%  Read in DF
try:
    aus_df = pd.read_csv(DATA_PATH)
except Exception:
    print("There was an error loading the CSV! Make sure it's in the right folder and try again.")
    quit()
categorical_features = aus_df.select_dtypes(include='object').columns.tolist()
numerical_features = aus_df.select_dtypes(include='number').columns.tolist()
dummy_categorical_features = [feat for feat in categorical_features if feat not in ['Location', 'Date']]
print(f"Number of Categorical Features: {len(categorical_features)}")
print(f"Number of Numerical Features: {len(numerical_features)}")
print(f"Time Range: {aus_df['Date'].min()} to {aus_df['Date'].max()}")
print(f"Columns:\n{aus_df.columns}")
print("***ORIGINAL***")
print(aus_df.head())

# %% Data Cleaning
def clean_data(aus_df):
    """
    Function for cleaning the dataframe
    :param aus_df: Australian Rain data
    :return: aus_df_clean: clean Australian Rain data
    """
    print("***Removing Duplicates***")
    print(f"Dropping {len(aus_df) - len(aus_df.duplicated(['Location']))} duplicates")
    aus_df = aus_df.drop_duplicates(subset=['Location', 'Date'], keep='first')

    # Handle binary Nulls - Drop these are worthless as they are primary data points
    print(f"{len(aus_df)} entries before dropping null binary categorical values")
    aus_df = aus_df.dropna(subset=['RainToday', 'RainTomorrow'])
    print(f"{len(aus_df)} entries after dropping null binary categorical values")

    # Fill Numercial Data with Mean and Categorical Data with Mode By Location
    print()
    print("***Filling data with mean values by location for numericals and mode values by location for categoricals***")
    print(f"Filling {aus_df.isnull().sum().sum()} nulls")
    locations = aus_df['Location'].unique()
    for location in locations:
        for column in aus_df.select_dtypes(include='number').columns:
            mean_val = aus_df.loc[aus_df['Location'] == location][column].mean()
            aus_df[column] = aus_df[column].fillna(mean_val, inplace=False)
        for column in aus_df.select_dtypes(include=['object']).columns:
            mode_val = aus_df.loc[aus_df['Location'] == location][column].mode()
            aus_df[column] = aus_df[column].fillna(mode_val, inplace=False)
    return aus_df

aus_df = clean_data(aus_df)
# Drop date now that duplicates are removed
aus_df = aus_df.drop("Date", axis=1)

# %% Data Analysis
aus_rain_numer = aus_df.select_dtypes(include='number')
# Run Pearson Correlation - Only get half the map
correlation = aus_rain_numer.corr(method='pearson', numeric_only=True)
matrix = np.triu(correlation)
sns.heatmap(correlation, annot=True, fmt='.2f', annot_kws={'fontsize': 'xx-small'}, linewidth=.5, mask=matrix)
plt.title('Correlation Matrix for Australia Rain Data')
plt.tight_layout()
plt.show()

# %% Feature Engineering
print("***Encoding Values***")
# Run Target encoding on Location based off a score created by other features
# Run One Hot Encoding for other variables
aus_df['Location_Enc'] = aus_df.groupby('Location')["Rainfall"].transform('mean')
aus_df = pd.get_dummies(
    aus_df,
    columns=dummy_categorical_features,
    drop_first=True, dtype='float')
print(f"Number of unique locations: {aus_df["Location"].nunique()}")
print(f"Number of unique encoded locations: {aus_df["Location_Enc"].nunique()}")

# Remove non-numerical datatypes
aus_df = aus_df.select_dtypes(include='number')

# %% Show preprocessed data
print(aus_df.info())

# %% Prepare dataset for Linear Regression - RUN THIS BEFORE CLASSIFICATION - USE THE VIF

aus_df_lin = aus_df.select_dtypes(include='number')
# Generate tomorrow rainfall column by shifting down one
MEAN_RAINFALL = aus_df_lin['Rainfall'].mean()
aus_df_lin['Rainfall_Tomorrow'] = aus_df_lin.groupby(['Location_Enc'])['Rainfall'].shift(-1, fill_value=MEAN_RAINFALL)
print(aus_df_lin[['Rainfall', 'Rainfall_Tomorrow']].head(20))
x_lin = aus_df_lin.drop(['Rainfall_Tomorrow', 'RainTomorrow_Yes'], axis=1)
x_lin[x_lin.columns] = MinMaxScaler().fit_transform(x_lin[x_lin.columns])
y_lin = aus_df_lin['Rainfall_Tomorrow']

print("Shape before VIF: ", x_lin.shape)

# Run collinearity test for linear data
vif_df = pd.DataFrame()
vif_df['variable'] = x_lin.columns
vif_df['VIF'] = [variance_inflation_factor(x_lin, i) for i in range(x_lin.shape[1])]
indices = np.argsort(vif_df['VIF'])

VIF_CUTOFF = 5
plt.figure(figsize=(10, 13))
plt.title('Variance Inflation Factor')
plt.barh(range(len(indices)), vif_df['VIF'], color='b', align='center')
plt.yticks(range(len(indices)), [vif_df.iloc[i]['variable'] for i in indices])
plt.xlabel('VIF')
plt.axvline(x=VIF_CUTOFF, linestyle='--', color='red', label='VIF Cutoff Threshold')
plt.legend()
plt.tight_layout()
plt.show()

needed_columns = ['RainToday_Yes', 'Rainfall']
drop_columns = [col for col in vif_df.loc[vif_df['VIF'] < VIF_CUTOFF, 'variable'].to_list() if col not in needed_columns]

x_lin = x_lin.drop(drop_columns, axis=1)

print("Shape after VIF: ", x_lin.shape)

X_lin_train, X_lin_test, Y_lin_train, Y_lin_test = train_test_split(
    x_lin, y_lin, test_size=0.2, random_state=RANDOM_SEED, shuffle=True)

# %% Linear Regression - Base Model
base_lin_model = sm.OLS(Y_lin_train, X_lin_train).fit()
print("Base Model Score: ", base_lin_model.rsquared_adj)
model_metric_map = {"Model Iteration": ["Base"],
                    "AIC": [base_lin_model.aic],
                    "BIC": [base_lin_model.bic],
                    "R-Squared Adj": [base_lin_model.rsquared_adj]
                    }

print("Base Model Regression Summary")
print(base_lin_model.summary())

# %% Linear Regression - Random Forest Analysis
# Run random Forest Analysis
IMPORTANCE_THRESHOLD = 0.01
rf_model = RandomForestRegressor(max_depth=10, random_state=RANDOM_SEED)
rf_endog = Y_lin_train
rf_exog = X_lin_train
rf_model.fit(rf_exog, rf_endog)
features = rf_exog.columns
importances = rf_model.feature_importances_
indices = np.argsort(importances)
plt.figure(figsize=(10, 10))
plt.title('Random Forest Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.axvline(x=IMPORTANCE_THRESHOLD, linestyle='--', color='red', label='Importance Threshold')
plt.legend()
plt.tight_layout()
plt.show()

important_features = [features[i] for i in indices if importances[i] >= IMPORTANCE_THRESHOLD]
print('***Important Features from Random Forest***')
print(important_features)

rf_lin_model = sm.OLS(Y_lin_train, X_lin_train[important_features]).fit()
print("Random Forest Model Score: ", rf_lin_model.rsquared_adj)
model_metric_map["Model Iteration"] += "Random Forest Model",
model_metric_map["AIC"] += rf_lin_model.aic,
model_metric_map["BIC"] += rf_lin_model.bic,
model_metric_map["R-Squared Adj"].append(rf_lin_model.rsquared_adj)

print("Random Forest Model Regression Summary")
print(rf_lin_model.summary())

# %% Linear Regression - Backwards Stepwise Regression
sfs = SequentialFeatureSelector(LinearRegression(),
                                direction='backward',
                                tol=0.05,
                                scoring='r2')
sfs.fit(X_lin_train, Y_lin_train)
selected_features = np.array(X_lin_train.columns)
print(
    "Features selected by backward sequential selection: "
    f"{selected_features[sfs.get_support()]}"
)

br_lin_model = sm.OLS(endog=Y_lin_train, exog=X_lin_train[selected_features]).fit()
print("Backwards Stepwise Regression Model Score: ", br_lin_model.rsquared_adj)
model_metric_map["Model Iteration"] += "Backwards Stepwise Regression Model",
model_metric_map["AIC"] += br_lin_model.aic,
model_metric_map["BIC"] += br_lin_model.bic,
model_metric_map["R-Squared Adj"].append(br_lin_model.rsquared_adj)


print("Backwards Stepwise Regression Summary")
print(br_lin_model.summary())

# %% Choose Linear Model
print("Final Regression Model Evaluation Table")
print(tabulate.tabulate(model_metric_map, headers='keys', tablefmt='fancy_grid'))

# %% Linear Regression - Final Model Evaluation
pred_vs_actual = pd.DataFrame()
pred_vs_actual['Actual'] = Y_lin_test
pred_vs_actual['Prediction'] = base_lin_model.predict(exog=X_lin_test)
pred_vs_actual.sort_index(inplace=True)
x_index = np.arange(1, len(Y_lin_test) + 1)

fig, axs = plt.subplots(2)
# Plot the prediction vs original
axs[0].plot(x_index, pred_vs_actual['Prediction'], color='b', linestyle='--')
axs[0].set_title('Predicted Rainfall')
axs[1].plot(x_index, pred_vs_actual['Actual'], color='orange', linestyle='--')
axs[1].set_title('Actual Rainfall')
for ax in axs.flat:
    ax.set(xlabel='Observation #', ylabel='Rainfall (mm)')
    # ax.grid(True)

fig.suptitle('Rainfall vs Prediction From Backwards Step Regression')

plt.tight_layout()
plt.show()

mse_back_reg = metrics.mean_squared_error(pred_vs_actual['Prediction'], pred_vs_actual['Actual'])
print("Mean Squared Error of Final Linear Regression Model: ", mse_back_reg)
print(base_lin_model.summary())
ALPHA = 0.05
print("Confidence Interval For Final Model: ", base_lin_model.conf_int(alpha=ALPHA))

# %% Prepare Dataset for Classification - Fix Imbalances and Split Data

X = aus_df.drop('RainTomorrow_Yes', axis=1)
Y = aus_df['RainTomorrow_Yes']

# Drop columns from VIF
X = X.drop(drop_columns, axis=1)

# Run SMOTE
sm = SMOTE(random_state=RANDOM_SEED)
X_res, Y_res = sm.fit_resample(X, Y)

rainfall_count_yes = Y_res.sum()
rainfall_count_no = len(Y_res) - rainfall_count_yes

# Plot distribution post-SMOTE
figure, axes = plt.subplots(2, figsize = (14, 9))
sns.countplot(x='RainTomorrow_Yes', data=Y.to_frame(), ax=axes[0], hue='RainTomorrow_Yes')
axes[0].set_title('RainTomorrow_Yes Distribution Pre-SMOTE')
axes[0].set_xlabel("Rain?")
axes[0].set_ylabel("Count")
sns.countplot(x='RainTomorrow_Yes', data=Y_res.to_frame(), ax=axes[1], hue='RainTomorrow_Yes')
axes[1].set_title('RainTomorrow_Yes Distribution Post-SMOTE')
axes[1].set_xlabel("Rain?")
axes[1].set_ylabel("Count")
figure.suptitle('Boolean Rain Values After SMOTE')
plt.grid()
plt.tight_layout()
plt.show()

# Split and standardize data
X_train, X_test, Y_train, Y_test = train_test_split(
    X_res, Y_res, test_size=0.2, random_state=RANDOM_SEED)
X_test = StandardScaler().fit_transform(X_test)
X_train = StandardScaler().fit_transform(X_train)

# Run PCA for dimension reduction
pca = PCA(random_state=RANDOM_SEED, svd_solver='full', n_components='mle')
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print("Old dimensionality: ", X_train.shape, X_test.shape, "New dimensionality: ", X_train_pca.shape, X_test_pca.shape)

# Define Cross Validation
cv = KFold(n_splits=7, shuffle=True, random_state=RANDOM_SEED)

# %% Classification Analysis - Define Evaluation Dictionary and Evaluation Function

eval_dict = {
    "Model": [],
    "Precision": [],
    "Recall": [],
    "Specificity": [],
    "F1-Score": [],
    "AUC ROC": []
}


def eval_model(model, model_name):
    """
    Function for evaluating a given model, returns classifiers and plots roc-auc curve and the confusion matrix
    :param model: The sklearn classifier
    :param model_name: The name of the model used
    """
    y_pred = model.predict(X_test_pca)
    ## ROC AUC
    prob = model.predict_proba(X_test_pca)
    prob = prob[:, 1]
    fper, tper, _ = metrics.roc_curve(Y_test, prob)
    roc_auc = metrics.roc_auc_score(Y_test, y_pred)

    plt.plot(fper, tper, label=f'{model_name} AUC = {roc_auc:0.2f}', color='blue', lw=2)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc=4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic (ROC)')
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot Confusion Matrix
    confusion_matrix = metrics.confusion_matrix(Y_test, y_pred)
    metrics.ConfusionMatrixDisplay(confusion_matrix).plot()
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

    # Calculate Metrics
    precision = metrics.precision_score(Y_test, y_pred)
    recall = metrics.recall_score(Y_test, y_pred)
    specificity = confusion_matrix[0] / (confusion_matrix[0] + confusion_matrix[1])
    f_score = metrics.f1_score(Y_test, y_pred)
    eval_dict["Model"].append(model_name)
    eval_dict["Precision"].append(precision)
    eval_dict["Recall"].append(recall)
    eval_dict["Specificity"].append(specificity)
    eval_dict["F1-Score"].append(f_score)
    eval_dict["AUC ROC"].append(roc_auc)


# %% Classification Analysis - Decision Tree Pre and Post Prune

# # Pre Pruning with Grid Search CV ONLY RUN IF NEED BE
# tree_parameters = {'max_depth': [None, 5, 10, 20, 30, 50],
#                     'min_samples_split': [2, 4, 10],
#                     'min_samples_leaf': [1, 2, 5],
#                     'criterion': ['gini', 'entropy'],
#                     'splitter': ['best', 'random'],
#                     'max_features': ['sqrt', 'log2']}
# grid_search = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_SEED), tree_parameters, scoring='accuracy', cv=cv)
# grid_search.fit(X_train_pca, Y_train)
# best_tree = grid_search.best_estimator_
# pprint.pprint(best_tree.get_params())

# Params gotten through above
# {'ccp_alpha': 0.054120742819807444m,
#  'class_weight': None,
#  'criterion': 'entropy',
#  'max_depth': None,
#  'max_features': 'sqrt',
#  'max_leaf_nodes': None,
#  'min_impurity_decrease': 0.0,
#  'min_samples_leaf': 1,
#  'min_samples_split': 2,
#  'min_weight_fraction_leaf': 0.0,
#  'monotonic_cst': None,
#  'random_state': 5805,
#  'splitter': 'best'}
#
# # Post Pruning - Cost Complexity Pruning
# post_prune_X, post_prune_Y = X_train_pca[1:1000, :], Y_train[1:1000]
# path = best_tree.cost_complexity_pruning_path(post_prune_X, post_prune_Y)
# ccp_alphas = path.ccp_alphas
#
# clfs = []
# for ccp_alpha in ccp_alphas:
#     clf = DecisionTreeClassifier(random_state=RANDOM_SEED, ccp_alpha=ccp_alpha)
#     clf.fit(post_prune_X, post_prune_Y)
#     clfs.append(clf)
# print(
#     "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
#         clfs[-1].tree_.node_count, ccp_alphas[-1]
#     )
# )
#
# train_scores = [clf.score(X_train_pca, Y_train) for clf in clfs]
# test_scores = [clf.score(X_test_pca, Y_test) for clf in clfs]
#
# fig, ax = plt.subplots()
# ax.set_xlabel("alpha")
# ax.set_ylabel("accuracy")
# ax.set_title("Accuracy vs alpha for training and testing sets")
# ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
# ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
# ax.legend()
# plt.show()

# %% Classification Analysis - Evaluate Decision Tree

# Post Pruning was causing an error - had to use zero for ccp alpha
best_tree = DecisionTreeClassifier(random_state=RANDOM_SEED, max_depth=None, max_features='sqrt', min_samples_split=2,
                                   min_samples_leaf=1, criterion='entropy', splitter='best',
                                   ccp_alpha=0)  # Refit using actual params
best_tree.fit(X_train_pca, Y_train)
eval_model(best_tree, "Decision Tree")

# %% Classification Analysis - Logistic Regression

# GRID SEARCH DO NOT RUN UNLESS NECCESSARY
# log_params = {
#     'l1_ratio': np.linspace(0, 1, 10),
#     'C': np.linspace(0.01, 100, 5),
# }
# grid_search = GridSearchCV(LogisticRegression(random_state=RANDOM_SEED, solver='saga', penalty='elasticnet', multi_class='ovr'), log_params, scoring='accuracy', cv=cv)
# grid_search.fit(X_train_pca, Y_train)
# print(grid_search.best_params_)
# best_log = grid_search.best_estimator_
# PARAMS GOTTEN:
# {'C': 25.0075, 'l1_ratio': 0.0}
best_log = LogisticRegression(random_state=RANDOM_SEED, solver='saga', penalty='elasticnet', multi_class='ovr',
                              C=25.0075, l1_ratio=0).fit(X_train_pca, Y_train)
eval_model(best_log, "Logistic Regression")

# %% Classification Analysis - KNN

# GRID SEARCH DO NOT RUN UNLESS NECCESSARY
# knn_params = {
#     "n_neighbors": [5, 10, 20, 30],
#     "p": [1, 2],
#     "weights": ["uniform", "distance"]
# }
# grid_search = GridSearchCV(KNeighborsClassifier(), knn_params, scoring='accuracy', cv=cv)
# grid_search.fit(X_train_pca, Y_train)
# best_knn = grid_search.best_estimator_
# print(grid_search.best_params_)
# Best Params
# {'n_neighbors': 5, 'p': 1, 'weights': 'distance'}
best_knn = KNeighborsClassifier(n_neighbors=5, p=1, weights='distance')
best_knn.fit(X_train_pca, Y_train)
eval_model(best_knn, "K-Nearest Neighbor")

# %% Classification Analysis - SVM

# DO NOT RUN - Too computationally expensive!
# svm_params = {
#     'kernel': ['linear', 'poly', 'rbf'],
# }
# grid_search = GridSearchCV(SVC(), svm_params, scoring='accuracy', cv=cv)
# grid_search.fit(X_train_pca, Y_train)
# best_svm = grid_search.best_estimator_
# print(grid_search.best_params_)
# best_svm = SVC(kernel='poly', probability=True, random_state=RANDOM_SEED, gamma='auto')
# best_svm.fit(X_train_pca, Y_train)
# # This is to dump the model because it takes a while to load
# with open(os.getcwd() + '/svm.pkl', 'wb') as f:
#     pickle.dump(best_svm, f)

# Read in model stored as pkl - Saves alot of time
try:
    with open('svm.pkl', 'rb') as f:
        best_svm = pickle.load(f)
except Exception:
    print("There was problem loading the pre-trained SVM Model. Making a new one. This is gonna take a while...")
    best_svm = SVC(kernel='poly', probability=True, random_state=RANDOM_SEED, gamma='auto')
    best_svm.fit(X_train_pca, Y_train)

eval_model(best_svm, "Support Vector Machine")

# %% Classification Analysis - Naive Bayes - Grid search is super quick
nb_params = {
    'var_smoothing': np.linspace(1e-9, 1e-5, 4)
}
grid_search = GridSearchCV(GaussianNB(), nb_params, scoring='accuracy', cv=cv)
grid_search.fit(X_train_pca, Y_train)
best_gnb = grid_search.best_estimator_
print(grid_search.best_params_)
eval_model(best_gnb, "Naive Bayes")

# %% Classification Analysis - Random Forest
# rf_params = {
#     'n_estimators': [100, 200, 300, 400],
#     'criterion': ['gini'],
#     'max_depth': [None, 10, 20, 30, 40, 50],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'bootstrap': [True, False],
#     'max_features': ['sqrt', 'log2']
# }
#
# grid_search = GridSearchCV(RandomForestClassifier(random_state=RANDOM_SEED), rf_params, scoring='accuracy', cv=cv)
# grid_search.fit(X_train_pca, Y_train)
# best_rf = grid_search.best_estimator_
# print(grid_search.best_params_)
# {'n_estimators': 400, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_depth': 50, 'criterion': 'gini', 'bootstrap': True}
best_rf = RandomForestClassifier(n_estimators=400, criterion='gini', random_state=RANDOM_SEED, max_depth=50,
                                 max_features='sqrt')
best_rf.fit(X_train_pca, Y_train)
eval_model(best_rf, "Random Forest")

# %% Classification Analysis - Neural Network
# mlp_params = {
#     "hidden_layer_sizes": [(100,), (50,), (150,)],
#     "solver": ["lbfgs", "adam"],
#     "activation": ["logistic", "relu"],
#     "alpha": [0.0001, 0.001],
#     "learning_rate": ["constant", "adaptive"],
# }
# grid_search = GridSearchCV(MLPClassifier(), mlp_params, scoring='accuracy', cv=cv)
# grid_search.fit(X_train_pca, Y_train)
# best_mlp = grid_search.best_estimator_
# print(grid_search.best_params_)
best_mlp = MLPClassifier(activation='relu', alpha=0.0001, learning_rate='adaptive')
best_mlp.fit(X_train_pca, Y_train)
eval_model(best_mlp, "Perceptron")

#%% Classification Analysis - Choose Final Model
print("***Final Regression Model Evaluation Table***")
print(tabulate.tabulate(eval_dict, headers='keys', tablefmt='fancy_grid'))


#%% Classification Analysis - Evaluate Final Model

y_test_predict = best_rf.predict(X_test_pca)
confusion_matrix = metrics.confusion_matrix(Y_test, y_test_predict)

# Plot classifiers and misclassifiers in a 2D projection
Y_test_array = Y_test.to_numpy()
markers = ['s', 's', 'o', 'o']
colors = ['red', 'blue', 'red', 'blue']

true_pos = [idx[0] for idx, y_pred in np.ndenumerate(y_test_predict) if y_pred and y_pred == Y_test_array[idx]]
true_neg = [idx[0] for idx, y_pred in np.ndenumerate(y_test_predict) if not y_pred and y_pred == Y_test_array[idx]]
false_pos = [idx[0] for idx, y_pred in np.ndenumerate(y_test_predict) if y_pred and y_pred != Y_test_array[idx]]
false_neg = [idx[0] for idx, y_pred in np.ndenumerate(y_test_predict) if not y_pred and y_pred != Y_test_array[idx]]
class_types = [f'true_pos={confusion_matrix[1,1]}',
               f'true_neg={confusion_matrix[0,0]}',
               f'false_pos={confusion_matrix[0, 1]}',
               f'false_neg={confusion_matrix[1, 0]}']

for cluster_idxs, marker, color, class_type in zip([true_pos, true_neg, false_pos, false_neg], markers,
                                            colors, class_types):
    # Plot the testing points and how they performed
    plt.scatter(
        np.take(X_test_pca, cluster_idxs, 0)[:, 0],
        np.take(X_test_pca, cluster_idxs, 0)[:, 1],
        c=color,
        edgecolors="k",
        marker=marker,
        label=class_type
    )

plt.xlabel('X[0]')
plt.ylabel('X[1]')
plt.grid()
plt.legend()
plt.title('2D Projection for Final Classification Model Analysis')
plt.tight_layout()
plt.show()



# %% Clustering - Prepare Dataset
## We want to find out what the weather is in each location

# Only need Location, Sunshine, and Daily Rainfall amount
aus_df_cluster = pd.read_csv(DATA_PATH)
aus_df_cluster = clean_data(aus_df_cluster)[['Location', 'Rainfall', 'Sunshine']]
aus_df_cluster = aus_df_cluster.groupby('Location').mean()

# %% Clustering Analysis - KMeans

sse = {}
silhoutte = {}
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000, random_state=RANDOM_SEED)
    kmeans.fit(aus_df_cluster)
    sse[k] = kmeans.inertia_  # Inertia: Sum of distances of samples to their closest cluster center
    labels = kmeans.predict(aus_df_cluster)
    silhoutte[k] = metrics.silhouette_score(aus_df_cluster, labels)
fig, axes = plt.subplots(2, figsize=(15, 15))
axes[0].plot(list(sse.keys()), list(sse.values()))
axes[0].set_xlabel("Number of cluster")
axes[0].set_ylabel("SSE")
axes[0].set_title("Elbow Method for KMeans")
axes[1].plot(list(silhoutte.keys()), list(silhoutte.values()))
axes[1].set_xlabel("Number of cluster")
axes[1].set_ylabel("Silhouette Coefficient")
axes[1].set_title("Silhouette method for KMeans")
plt.show()

# Based on above, use k=5
# Cluster
kmeans = KMeans(n_clusters=5)

cluster_prediction = kmeans.fit_predict(aus_df_cluster)
centroids = kmeans.cluster_centers_
graph_labels = np.unique(cluster_prediction)

# Plot clustering
locations = aus_df_cluster.index
plt.figure(figsize=(10, 10))
# plotting the results:
for i in graph_labels:
    plt.scatter(aus_df_cluster.iloc[cluster_prediction == i]['Sunshine'],
                aus_df_cluster.iloc[cluster_prediction == i]['Rainfall'],
                label=i)

plt.scatter(centroids[:, 1], centroids[:, 0], marker="x", s=150, linewidths=5, zorder=10, label='Centroid')

for i, location in enumerate(locations):
    plt.annotate(location, (aus_df_cluster.iloc[i]['Sunshine'],
                            aus_df_cluster.iloc[i]['Rainfall']),
                 fontsize='x-small')

plt.xlabel('Sunshine', fontweight='bold', horizontalalignment='center')
plt.ylabel('Rainfall', fontweight='bold', horizontalalignment='center')
plt.legend()
plt.grid()
plt.title('Rainy Vs Sunny Locations in Australia', fontweight='bold')
plt.tight_layout()
plt.show()

# %% Association Rule mining - Apriori
# Convert all numerical columns to categorical using mean
aus_df_association = clean_data(pd.read_csv(DATA_PATH)).drop(['Location', 'Date', 'Rainfall', 'WindDir3pm', 'WindDir9am', 'WindGustDir'], axis=1)
numerical_features = aus_df_association.select_dtypes(include='number')
for feature in numerical_features:
    aus_df_association[feature] = pd.qcut(aus_df_association[feature], q=2, labels=[False, True])

# Convert yes/no to True/False
aus_df_association['RainToday'] = aus_df_association['RainToday'].map({"Yes": True, "No": False})
aus_df_association['RainTomorrow'] = aus_df_association['RainTomorrow'].map({"Yes": True, "No": False})

print(aus_df_association.head())

apriori_df = apriori(aus_df_association, min_support=0.4, use_colnames=True)
apriori_df['length'] = apriori_df['itemsets'].apply(lambda x: len(x))
apriori_df = apriori_df[apriori_df['length'] >= 2]
print("***Association Rule Mining Table (Apriori Algorithm)***")
print(tabulate.tabulate(apriori_df[['support', 'itemsets']], headers='keys', tablefmt='grid'))