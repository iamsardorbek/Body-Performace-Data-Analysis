# main program that:
# - holds EDA removing duplicates, checking for data inconsistencies, encoding categorical values into numbers and
# visualizing the data
# - trains a classifier XGBoost model and computes the accuracy of the model

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy.stats import loguniform
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split

# Before starting work, I reviewed the csv dataset and noticed that column names were non-standard. This would create
# inconveniences referencing them in code, so I renamed some columns to use '_' instead of spaces.

if __name__ == '__main__':
    data = pd.read_csv("bodyPerformanceDataset.csv")

    # 1. understand the data and save the results in a text file "results.txt" (for readability)
    results_file = open(r"results.txt", "w")

    results_file.write("Primary information about the dataset:\n")
    results_file.write("Total number of rows: " + str(data.shape[0]) + "\n")
    results_file.write("Total number of columns: " + str(data.shape[1]) + "\n")
    results_file.write("Names of the columns: ")
    results_file.write(', '.join(data.columns) + "\n")
    results_file.write("Number of duplicate rows: " + str(data.duplicated().sum()) + "\n")
    results_file.write("\n")

    # remove duplicates if there are any
    if data.duplicated().sum() != 0:
        data.drop_duplicates()

    results_file.write("Statistical information about the dataset:\n")
    results_file.write(data.describe().to_string())
    results_file.write("\n\n")

    results_file.write("Heading 5 rows of the dataset:\n")
    results_file.write(data.head().to_string())
    results_file.write("\n\n")

    results_file.write("Tailing 5 rows of the dataset:\n")
    results_file.write(data.tail().to_string())
    results_file.write("\n\n")

    results_file.write("Number of unique values in each column:\n")
    results_file.write(data.nunique().to_string())
    results_file.write("\n\n")

    results_file.write("Unique values in column 'class': ")
    class_values = data['class'].unique()
    results_file.write(', '.join(class_values.tolist()))
    results_file.write("\n\n")

    # check data integrity
    rows_with_wrong_age = data[data.age < 0]
    rows_with_wrong_height = data[data.height_cm <= 0]
    rows_with_wrong_weight = data[data.weight_kg <= 0]
    rows_with_wrong_body_fat_percentage = data[data.body_fat_percentage <= 0]
    rows_with_wrong_diastolic = data[data.diastolic <= 0]
    rows_with_wrong_systolic = data[data.systolic <= 0]
    rows_with_wrong_gripForce = data[data.gripForce < 0]
    rows_with_wrong_sit_and_bend_forward = data[data.sit_and_bend_forward_cm <= 0]
    rows_with_wrong_sit_ups_count = data[data.sit_ups_count < 0]
    rows_with_wrong_broad_jump = data[data.broad_jump_cm <= 0]

    # 2. clean the data
    if not rows_with_wrong_age.empty:
        results_file.write("Rows with wrong age:\n")
        results_file.write(rows_with_wrong_age.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_age.index, inplace=True)
        results_file.write("Rows with wrong age deleted.\n\n")
    if not rows_with_wrong_height.empty:
        results_file.write("Rows with wrong height:\n")
        results_file.write(rows_with_wrong_height.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_height.index, inplace=True)
        results_file.write("Rows with wrong height deleted.\n\n")
    if not rows_with_wrong_weight.empty:
        results_file.write("Rows with wrong weight:\n")
        results_file.write(rows_with_wrong_weight.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_weight.index, inplace=True)
        results_file.write("Rows with wrong weight deleted.\n\n")
    if not rows_with_wrong_body_fat_percentage.empty:
        results_file.write("Rows with body fat percentage:\n")
        results_file.write(rows_with_wrong_body_fat_percentage.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_body_fat_percentage.index, inplace=True)
        results_file.write("Rows with wrong body fat percentage deleted.\n\n")
    if not rows_with_wrong_diastolic.empty:
        results_file.write("Rows with wrong diastolic:\n")
        results_file.write(rows_with_wrong_diastolic.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_diastolic.index, inplace=True)
        results_file.write("Rows with wrong diastolic deleted.\n\n")
    if not rows_with_wrong_systolic.empty:
        results_file.write("Rows with wrong systolic:\n")
        results_file.write(rows_with_wrong_systolic.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_systolic.index, inplace=True)
        results_file.write("Rows with wrong systolic deleted.\n\n")
    if not rows_with_wrong_gripForce.empty:
        results_file.write("Rows with wrong grip force:\n")
        results_file.write(rows_with_wrong_gripForce.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_gripForce.index, inplace=True)
        results_file.write("Rows with wrong grip force deleted.\n\n")
    if not rows_with_wrong_sit_and_bend_forward.empty:
        results_file.write("Rows with wrong sit and bend forward length:\n")
        results_file.write(rows_with_wrong_sit_and_bend_forward.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_sit_and_bend_forward.index, inplace=True)
        results_file.write("Rows with wrong sit and bend forward length deleted.\n\n")
    if not rows_with_wrong_sit_ups_count.empty:
        results_file.write("Rows with wrong sit-ups count:\n")
        results_file.write(rows_with_wrong_sit_ups_count.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_sit_ups_count.index, inplace=True)
        results_file.write("Rows with wrong sit-ups count deleted.\n\n")
    if not rows_with_wrong_broad_jump.empty:
        results_file.write("Rows with wrong broad jump length:\n")
        results_file.write(rows_with_wrong_broad_jump.to_string())
        results_file.write("\n")
        data.drop(rows_with_wrong_broad_jump.index, inplace=True)
        results_file.write("Rows with wrong broad jump length deleted.\n\n")

    # change categorical values of gender ('F', 'M') to binary (0, 1) values
    data["gender"] = np.where(data["gender"] == "F", 0, 1)
    data["class"] = np.where(data["class"] == "A", 0, data["class"])
    data["class"] = np.where(data["class"] == "B", 1, data["class"])
    data["class"] = np.where(data["class"] == "C", 2, data["class"])
    data["class"] = np.where(data["class"] == "D", 3, data["class"])

    # 3. relationship analysis
    # correlation heatmap
    correlation = data.corr()
    sns.heatmap(correlation, xticklabels=correlation.columns, yticklabels=correlation.columns, annot=True)
    plt.savefig("correlation_heatmap.png")  # save to a file
    image = cv2.imread('pair_plot.png')
    cv2.imshow('pair_plot', image)
    # calculate and output the correlation data to results.txt
    correlation_abs = correlation.abs()
    np.fill_diagonal(correlation_abs.values, 0)
    correlation_abs_max = correlation_abs.max().sort_values(ascending=False)[0:2]
    correlation_abs_max = correlation_abs_max.to_dict()
    column_names = list(correlation_abs_max.keys())
    results_file.write("Max correlation is between %s and %s, absolute value = %f\n\n"
                       % (column_names[0], column_names[1], correlation_abs_max[column_names[0]]))

    # pair plot for all columns of the dataset
    # fyi: this plot is big for the given dataset
    sns.pairplot(data)
    plt.savefig("pair_plot.png")
    plt.clf()
    image = cv2.imread('pair_plot.png')
    cv2.imshow('pair_plot', image)

    # relational plot with three vars: height, weight and gender
    sns.relplot(x='height_cm', y='weight_kg', hue='gender', data=data)
    plt.savefig("relational_plot.png")  # save to a file
    plt.clf()
    image = cv2.imread('relational_plot.png')
    cv2.imshow('relational_plot', image)

    # histogram which display the distribution of "sit-ups count" values
    sns.histplot(data['sit_ups_count'])
    plt.savefig("histogram_plot.png")  # save to a file
    plt.clf()
    image = cv2.imread('histogram_plot.png')
    cv2.imshow('histogram_plot', image)

    # box plot to observe the main statistical insights about body fat percentage values
    sns.catplot(x='body_fat_percentage', kind='box', data=data)
    plt.savefig("categorical_plot.png")  # save to a file
    plt.clf()
    image = cv2.imread('categorical_plot.png')
    cv2.imshow('categorical_plot', image)
    results_file.write("You can check out the visualizations made with seaborn and matplotlib tools in the following "
                       "image files: correlation_heatmap.png, pair_plot.png, relational_plot.png, "
                       "histogram_plot.png, categorical_plot.png\n\n")

    # train a classification model, because the output data is discrete (A, B, C, D) rather than continuous.
    # We would use regression model for that case

    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=7)
    model = xgb.XGBClassifier()
    model.fit(X_train, y_train)
    model.save_model("model.json")
    predictions = model.predict(X_test)
    predictions = list(predictions)
    y_test = list(y_test)
    accuracy = accuracy_score(y_test, predictions)

    # save the confusion matrix of the dataset to "confusion_matrix.png"
    cm = confusion_matrix(y_test, predictions)
    df_cm = pd.DataFrame(cm, ['A', 'B', 'C', 'D'], ['A', 'B', 'C', 'D'])
    sns.set(font_scale=1.25)  # for label size
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12})  # font size
    plt.savefig("confusion_matrix.png", bbox_inches='tight')  # save to a file
    plt.clf()

    # classification report
    clf_report = classification_report(y_test, predictions, target_names=['A', 'B', 'C', 'D'], output_dict=True)
    sns.set(font_scale=1)  # for label size
    sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    plt.savefig("classification_report.png", bbox_inches='tight')  # save to a file
    plt.clf()

    # Result is displayed in an openCV window too
    # Setup text display parameters
    text_accuracy = "Accuracy Computed: %f" % accuracy
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_position_accuracy = (10, 400)
    font_scale = 1
    font_color = (255, 255, 255)
    thickness = 1
    line_type = 2

    result_img = np.zeros((512, 512, 3), np.uint8)  # Create a black image
    cv2.putText(result_img, text_accuracy,
                text_position_accuracy, font, font_scale, font_color, thickness, line_type)
    cv2.imshow("Result of testing the classification model", result_img)

    results_file.write("XGBoost classifier model has been trained to predict 'class' of a person.\n"
                       "The accuracy of the model is %d percent\n" % (accuracy * 100))
    results_file.write("\n\n")

    results_file.close()

    # Optimization techniques for parameter tuning. RandomizedSearchCV
    # define search space
    space = dict()
    space['solver'] = ['newton-cg', 'lbfgs', 'liblinear']
    space['penalty'] = ['none', 'l1', 'l2', 'elasticnet']
    space['C'] = loguniform(1e-5, 100)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    # the below function is taking long to execute fully
    search = RandomizedSearchCV(model, space, n_iter=10, scoring='accuracy', n_jobs=-1, cv=cv, random_state=1)
    result = search.fit(X, list(y))
    print('Best Score: %s' % result.best_score_)
    print('Best hyper parameters: %s' % result.best_params_)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
