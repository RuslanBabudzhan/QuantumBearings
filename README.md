# QuantumBearings

**QuantumBearings** - project the essence of which is the detection of defective rolling bearings with machine learning methods based on bearings acceleration data. 

Project includes data collection using a device that simulates bearings` work, processing of the received data and signals classification for recognizing defective bearings.

Most of the features for working with data can be seen in this [Usage examples notebook](https://nbviewer.jupyter.org/github/RuslanBabudzhan/QuantumBearings/blob/master/notebooks/UsageExamples.ipynb "Usage examples").



### Table of Contents
_under construction_

### Data mining
To create dataset, a test bench has been developed and configured to simulate the operation of the rotor system. Vibration sensors have been used to monitor the state of mechanisms in an automatic mode, to classify the quality of bearings operation with machine learning methods.

<details>
<summary><b>Test bench and experiments conducting details</b></summary>

<img src="experiments/images/ReadMe/drawing.png" alt="drawing" width="600"/>
<img src="experiments/images/ReadMe/test bench.png" alt="test bench" width="600"/>

The control unit is powered by a standard current of 220 V. Inside the control unit there is a 30 V power supply powering a motor.
External control comes from the Arduino. At the input, the Arduino receives the rotation speed, and at the output it supplies a PWM signal, the frequency of which is generated by the PID controller. The regulator has been tuned and calibrated so that the motor accelerates to 1500 rpm, then maintains this speed and then also slowly decelerates to 200 rpm. This has been done so that the experimental procedure is similar to each other for any bearing. Thus, for the further analysis it is proposed to use a stationary time section of the installation with a constant shaft rotation frequency. This is a section with an interval of 10 to 20 s.

Before experiments the bearings have been mounted on the shaft. Bearing on position 1 is constant during all experiments. This bearing is new, purchased before starting the experiments. Bearings on position 2 have been previously used in various workbenches and machines and have been replaced from one experiment to another. In this way, the device and feature generation methods aim to classify the bearings on position 2.

Data collection has been performed with a sampling rate of 3000 records per second. We tested 2 types of bearings - 6202 and 6204 deep groove bearings. 

**Тhe number of bearings of the investigated types:**

|      Type      | 6204 | 6202 |
|:--------------:|:----:|:----:|
|  with defect   |  30  |  70  |
| without defect |  7   |  5   |

The data was collected according to the acceleration-hold-stop scheme. First, the rotor was accelerated to the desired speed. Then there was a 10-second hold (hereinafter the stationary interval). Then the motor stopped. The recording was carried out for the full load interval.
****
</details>

The resulting dataset consists of 10265700 recordings that describe rotors behavior, 91600 per bearing on average. Collecting data has been uploaded on platform Kaggle and it is in the public domain ([link](https://www.kaggle.com/isaienkov/bearing-classification)). Detailed information about resulting dataset is presented below. For classification, the collected acceleration data of bearings in three axes: X, Y, Z will be used. The name of these features contains the bearing index and the acceleration axis.

<img src="experiments/images/ReadMe/signal.png" alt="Signals" width="1000"/>


|     Field      | Description                                                    | Units |
|:--------------:|:---------------------------------------------------------------|:-----:|
| Experiment ID	 | Unique identifier of the experiment                            |   -   |
| Bearing 1/2 ID | 	Unique identifier of the bearing on the first/second position |  	-   |
|   Timestamp	   | Time, measured in seconds                                      | 	sec  |
|    A1_X/Y/Z    | 	Acceleration along the X, Y and Z axes for the first bearing  | m/s2  |
|    A2_X/Y/Z    | 	Acceleration along the X, Y and Z axes for the second bearing | m/s2  |
|     RPM/HZ     | 	Rotation speed                                                |  rpm  |
|       W	       | The motor power at a time                                      | Watts |

We also looked at the possibility of using off-the-shelf datasets. So far we have looked at two third-party datasets:
- First César Ricardo Soto-Ocampo et al. dataset ([link](https://zenodo.org/record/3898942#.YYwVSWDP2Um "Cesar 1")). Consists 45 bearings. Further referred to as Cesar №1
- Second César Ricardo Soto-Ocampo et al. dataset ([link](https://zenodo.org/record/5084405#.YYwVSWDP2Ul "Cesar 2")). Consists 45 bearings. Further referred to as Cesar №2

Both datasets presented as *.mat types. We have converted these datasets into our format as follows:

```python
from source.preprocessing.converter import Converter


mat_files_path = 'N1 Cesar Ricardo'
data = Converter.cesar_convert(mat_files_path)
```

### Data processing
_under construction_

### Building models

An example of tuning the models is available in this [GridSearch Notebook](https://nbviewer.jupyter.org/github/RuslanBabudzhan/QuantumBearings/blob/master/notebooks/GridSearch.ipynb "Usage examples").

Due to the very small amount of data, we settled on using classic machine learning models. We used such models:
1. Logistic Regression
2. Support Vector Classifier
3. Random Forest
4. K-Nearest Neighbours

We have a strong class imbalance in our dataset - 100 negative and 12 positive instances. Therefore, we use [F1 metric](https://en.wikipedia.org/wiki/F-score) as the main quality metric. We also look at other metrics such as precision and TPR. The metrics available in the project can be found as follows:
```python
from source.datamodels.iterators import Metrics


Metrics.get_keys()
```
```
['accuracy', 'precision', 'recall', 'f1', 'TPR', 'TNR']
```

Since there are only 112 records in our dataset, the quality of the models is highly dependent on splitting the sample into train and test subsamples. We should not use data from the same batch in both the training and the test set (the models must be able to recognize bearing signals that have not been encountered before), so we decided to use bootstrapped samples for training. Thus, we can generate an infinitely large number of values of the target metric, and average its values in order to have a stable estimate of the quality of the model.

In our work, we use scikit-learn machine learning library to build models. Thus, we can use a ready-made method for tuning the models ([Usage example](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html "sklearn GridSearch")). But scikit-learn does not provide the ability to generate grouped train and test samples with overlaps. So, we have created a custom indices generator for our task:
```python
import numpy as np
from source.processes.Shuffler import OverlapGroupCV


X = np.arange(32).reshape((8, 4))  # Dataset with 8 rows
y = np.arange(2).repeat(4)  # targets
groups = np.arange(4).repeat(2)  # group label for each row in X

cv = OverlapGroupCV(train_size=0.5, n_repeats=100).split(X, y, groups)

train_subset_indices, test_subset_indices = next(cv)

print(f"Train subset indices: {train_subset_indices}\nTest subset indices: {test_subset_indices}\n")
print(f"Train subset: \n{X[train_subset_indices]}\n\nTest subset: \n{X[test_subset_indices]}")
```
```
Train subset indices: [0, 1, 4, 5]
Test subset indices: [2, 3, 6, 7]

Train subset: 
[[ 0  1  2  3]
 [ 4  5  6  7]
 [16 17 18 19]
 [20 21 22 23]]

Test subset: 
[[ 8  9 10 11]
 [12 13 14 15]
 [24 25 26 27]
 [28 29 30 31]]
```
Note that for proper work it is necessary that the labels for all elements from one group are the same. Otherwise, it will shoot your ass.

We tested several types of dataset preprocessing. For our dataset, we considered the following options:
1. Use both signal and spectrum of all 6 axes and all 22 statistics;
2. Use statistics of signals along all 6 axes;
3. Use statistics of spectrum along all 6 axes;
4. Use both signal and spectrum statistics along Y axis.

For all options, we considered signals from both the mounted and replaceable bearing.

<details>
<summary><b>Best results for models tuning with DSM dataset</b></summary>

The best option was to train the model on signal statistics. Best F1 score: 84%
<img src="experiments/images/ReadMe/bar_DSM_GS_AXYZ_signal_22stats_6_11_2021_4_11_2021.png" alt="GridSearch Bar" width="900"/>
<img src="experiments/images/ReadMe/kde_DSM_GS_AXYZ_signal_22stats_6_11_2021_4_11_2021.png" alt="GridSearch KDE" width="900"/>
<img src="experiments/images/ReadMe/box_DSM_GS_AXYZ_signal_22stats_6_11_2021_4_11_2021.png" alt="GridSearch Box" width="900"/>
</details>

We also looked at a third party dataset (Cesar №2). We trained the models on this set, using different signal scaling before splitting into batches. So, for this dataset we considered the following options:
1. Pure signal, without scaling;
2. Standard scale (z-score);
3. MinMax scale ([link](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html "Robust scaler"));
4. Robust scale ([link](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html "Robust scaler")).

You can see all the results for both datasets [here](https://github.com/RuslanBabudzhan/QuantumBearings/experiments/ResultTables/public/ "Result tables"). The charts are [here](https://github.com/RuslanBabudzhan/QuantumBearings/experiments/images/public/GridSearch "GridSearch").
### Compare datasets
_under construction_

### Feature selection
_under construction_

### Results
To find the best way to predict bearing condition, we tested a several number of models and datasets - there are currently 96 options. Since we are using bootstrap sampling, we want to store all the information about our experiments and have full reproducibility to significantly save computational time. Therefore, we decided to save the results of experiments through the [pydantic](https://pydantic-docs.helpmanual.io/) library to save the results into data models and serialize them to drive. All data models listened in ```source.datamodels.datamodels```.
Data models provides information about datasets, model, train and test indices, predictions, metrics etc.   

Currently available results` data models:

- Single run. For single fit of one model.
- Single datasets comparison. Single model fit with training on one dataset and testing on another.
- Bootstrap run. Model fit using `OverlapGroupCV` instance to create a distribution of metrics.
- Bootstrap run with feature selection.
- GridSearch. Bootstrap with model tuning.
- Bootstrap dataset comparison results. To create metrics distribution with training on subsamples of one dataset and testing on subsamples of another.

To compare the quality of different models and different approaches, we created tables with the results of the experiments. They are automatically generated from the serialized results and contain information about the approach to processing the dataset and estimated scores. All results are available in [this folder](https://github.com/RuslanBabudzhan/QuantumBearings/experiments/ResultTables/public "public results").

An example of creating such a table:
```python
from source.datamodels.datamodels import GridSearchResults
from source.postprocessing.mlcsv import generate_csv_from_results, create_readable_xlsx
from source.postprocessing.mljson import deserialize_results


results = deserialize_results(result_obj_type=GridSearchResults, filepath='GridSearch')  # deserialize the results that are in the "GridSearch" directory

generate_csv_from_results(results=results, csv_name="DSM_GS.csv", results_type=GridSearchResults)  # generate csv file. We use csv
# for intermediate saving of tables, as well as in order to be able to add new records to the existing table

create_readable_xlsx(xlsx_name="DSM GridSearch results.xlsx", csv_name="DSM_GS.csv")  # create ML experiment tracking table
```
<details>
<summary><b>Example of ML experiment tracking table</b></summary>
This table contains data about tuning models when training on our dataset.

| experiment index | run_label | model_name | use_signal | use_specter | specter_threshold | resampling_number | Scores: accuracy | Scores: precision | Scores: recall | Scores: f1 | Scores: TPR | Scores: TNR |
|:----------------:|-----------|------------|------------|-------------|-------------------|-------------------|------------------|-------------------|----------------|------------|-------------|-------------|
|        0         | GS        | RFC        | Yes        | Yes         | 500               | 100               | 0,953            | 0,991             | 0,613          | 0,747      | 0,613       | 0,999       |
|        1         | GS        | LR         | Yes        | Yes         | 500               | 100               | 0,957            | 0,908             | 0,721          | 0,792      | 0,721       | 0,988       |
|        2         | GS        | SVC        | No         | Yes         | 500               | 100               | 0,909            | 0,667             | 0,49           | 0,547      | 0,49        | 0,966       |
|        3         | GS        | LR         | No         | Yes         | 500               | 100               | 0,908            | 0,655             | 0,489          | 0,541      | 0,489       | 0,965       |
|        4         | GS        | KNN        | Yes        | Yes         | 500               | 100               | 0,921            | 0,718             | 0,57           | 0,624      | 0,57        | 0,968       |
|        5         | GS        | KNN        | Yes        | Yes         | 500               | 100               | 0,954            | 0,896             | 0,71           | 0,776      | 0,71        | 0,987       |
|        6         | GS        | RFC        | Yes        | Yes         | 500               | 100               | 0,956            | 0,992             | 0,634          | 0,761      | 0,634       | 0,999       |
|        7         | GS        | RFC        | Yes        | No          | 500               | 100               | 0,957            | 0,999             | 0,644          | 0,773      | 0,644       | 1,0         |
|        8         | GS        | SVC        | Yes        | Yes         | 500               | 100               | 0,951            | 0,94              | 0,627          | 0,739      | 0,627       | 0,994       |
|        9         | GS        | SVC        | Yes        | No          | 500               | 100               | 0,964            | 0,962             | 0,733          | 0,826      | 0,733       | 0,996       |
|        10        | GS        | SVC        | Yes        | Yes         | 500               | 100               | 0,961            | 0,959             | 0,703          | 0,804      | 0,703       | 0,995       |
|        11        | GS        | KNN        | No         | Yes         | 500               | 100               | 0,909            | 0,661             | 0,508          | 0,561      | 0,508       | 0,963       |
|        12        | GS        | LR         | Yes        | Yes         | 500               | 100               | 0,943            | 0,838             | 0,67           | 0,728      | 0,67        | 0,98        |
|        13        | GS        | LR         | Yes        | No          | 500               | 100               | 0,959            | 0,822             | 0,882          | 0,838      | 0,882       | 0,97        |
|        14        | GS        | RFC        | No         | Yes         | 500               | 100               | 0,91             | 0,802             | 0,345          | 0,462      | 0,345       | 0,986       |
|        15        | GS        | KNN        | Yes        | No          | 500               | 100               | 0,953            | 0,893             | 0,703          | 0,774      | 0,703       | 0,986       |

</details>

