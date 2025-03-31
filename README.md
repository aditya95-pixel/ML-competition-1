# Water Consumption Prediction Using CatBoost

## Approach
The objective of this project is to predict water consumption based on historical data using the CatBoost regressor. The approach involves preprocessing timestamp data, handling missing values, and optimizing hyperparameters to achieve the best predictive performance.

## Feature Engineering
1. **Timestamp Processing:**
   - Converted timestamps to datetime format.
   - Extracted features such as Hour, DayOfWeek, Month, DayOfMonth, and Year.
   - Created an `IsWeekend` feature to differentiate weekends from weekdays.
   - Applied sinusoidal transformations to cyclical features (Hour, DayOfWeek, and Month) to capture their periodic nature.

2. **Handling Missing Values:**
   - Used `SimpleImputer` with a median strategy for numerical features.
   - Applied `SimpleImputer` with the most frequent strategy for categorical features.

3. **Data Splitting:**
   - Sorted the dataset by Timestamp to maintain chronological order.
   - Used an 80-20 train-validation split.

## Model Selection and Hyperparameter Tuning
- **CatBoost Regressor** was chosen due to its effectiveness with categorical features and handling of missing values.
- A randomized search approach was used for hyperparameter tuning.
- The following hyperparameters were optimized over 20 iterations:
  - Learning rate: `[0.01, 0.03, 0.05, 0.1]`
  - Depth: `[4, 6, 8, 10]`
  - Iterations: `[500, 1000, 1500]`
  - L2 leaf regularization: `[1, 3, 5, 7]`
- The model was evaluated using Mean Squared Error (MSE), with a final score transformation: `score = max(0, 100 - sqrt(MSE))`.

## Tools Used
- **Python Libraries:**
  - `pandas`: Data manipulation and preprocessing.
  - `numpy`: Mathematical operations and transformations.
  - `scikit-learn`: Data splitting, imputations, and MSE calculation.
  - `CatBoost`: Gradient boosting model for regression.
  - `random`: Hyperparameter selection.

## Source Files
- `train.csv`: Training dataset.
- `test.csv`: Testing dataset.
- `model.py`: Contains the main code.
- `submission.csv`: The output file containing predictions for water consumption.

## Execution Steps
1. Read and preprocess the dataset.
2. Extract and engineer relevant features.
3. Handle missing values appropriately.
4. Split data into training and validation sets.
5. Tune hyperparameters using randomized search.
6. Train the final CatBoost model with the best parameters.
7. Generate predictions and save results.
