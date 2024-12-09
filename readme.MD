### ABSTRACT ### 

Accurate rainfall prediction is critical for agriculture, water resource management, and disaster mitigation,
particularly in regions like Australia, which experiences significant climatic variability. This study explores the
application of machine learning techniques to predict rainfall, leveraging historical meteorological data from
Australian weather stations. Several models, including logistic regression, decision trees, random forests, and neural
network approaches, were evaluated for their predictive accuracy. Key meteorological variables such as temperature,
humidity, wind speed, and atmospheric pressure were used as inputs. Feature selection and preprocessing techniques were
employed to address data imbalances and enhance model performance. Results indicate that ensemble methods, particularly
random forests, achieved superior accuracy and reliability, outperforming traditional statistical methods. This research
demonstrates the potential of machine learning to provide robust rainfall forecasts, contributing to more informed
decision-making in weather-sensitive sectors. Future work includes integrating real-time data and expanding the scope to
account for extreme weather events.

### HOW TO RUN ###

To run this code successfully, change the "DATA_PATH" global variable at the top of the python file to the location the
python file is stored (if stored in the same folder, it should not need to be changed). Additionally, there is an
svm.pkl which contains the trained SVC model, which should
SIGNIFICANTLY reduce runtime of the entire program if loaded properly.
This code should not need any additional libraries beyond what is standard to run it.

All Grid Searches have been commented out with results hardcoded to cut down on total runtime.