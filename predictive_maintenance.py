import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime
from fpdf import FPDF
import joblib
import logging

sns.set(style="whitegrid")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def phase1(): # Data Ingestion & Preprocessing
    # Adds column headers to the dataset
    # 'unit' - engine ID, 'time' - cycle count, 'op1'-;op3' - operational settings, 's1'-'s21' - sensor readings
    column_names = ['unit', 'time', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1,22)]
    df = pd.read_csv("CMaps/train_FD001.txt", sep=r'\s+', header=None, names=column_names) # NASA data is whitespace separated so sep= splits that data at the whitespace

    # Entire dataset is "grouped by" the unit (individual engine)
    # Within each group (engine), we specify to 'time' column
    # Then, we find the max time before engine failure and broadcast it over every row of the group
    max_cycles = df.groupby('unit')['time'].transform('max') # Returns a series where every row corresponding to that engine shows the final cycle count for that specific engine
    df['RUL'] = max_cycles - df['time'] # Remaining Useful Life (RUL) = Max Cycle Number for that unit - Current Cycle Number

    # Visualization of some random sensors (sensor #2 and sensor #5) of random engines (engine #1, engine #15, and engine #70) just for a feel of things
    for engine_id in [1, 15, 70]:
        subset = df[df['unit'] == engine_id] # Creates a mini dataframe with the information from engines 1, 15, and 70
        plt.figure(figsize=(12,4))
        plt.plot(subset['time'], subset['s2'], label='Sensor 2') # Plots a Time vs. Sensor 2 Readings graph
        plt.plot(subset['time'], subset['s5'], label='Sensor 5') # Plots a Time vs. Sensor 5 Readings graph
        plt.title(f"Engine {engine_id} - Sensor Readings")
        plt.xlabel("Time (Cycles)")
        plt.ylabel("Sensor Value")
        plt.legend()
        plt.tight_layout()
        plt.show()
        # Sensors 2 and 5 are contant throughout their runtime for multiple engines with near-zero variance. => Contribute little to no useful information. => Uninformative/Flat sensors
    
    # Learns the mean and standard deviation of the input data and normalizes all the data with that info
    # Only scale operational columns and not categorical columns
    cols_to_scale = ['op1', 'op2', 'op3'] + [f's{i}' for i in range(1,22)]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cols_to_scale])
    joblib.dump(scaler, "scaler.pkl")
    df_scaled_sensors = pd.DataFrame(scaled_data, columns=cols_to_scale)
    # Combine scaled sensors with the rest of the columns (unit, time, RUL)
    df_scaled = pd.concat([df[['unit', 'time', 'RUL']].reset_index(drop=True), df_scaled_sensors.reset_index(drop=True)], axis=1)

    return df_scaled


def phase2(df_scaled): # Feature Engineering
    # Loops through all 21 sensors and finds the rolling mean and rolling standard deviation for each of them
    for i in range(1,22):
        sensor = f's{i}'
        # .groupby('unit') - breaks the dataset into separate mini-DataFrames for each engine
        # lambda - a function within one line
        # .transform() - a function that applies another function to each subgroup then stitches the results back together to match full DataFrame’s index
        df_scaled[f'{sensor}_roll_mean'] = df_scaled.groupby('unit')[f'{sensor}'].transform(lambda x: x.rolling(window=5).mean())
        df_scaled[f'{sensor}_roll_std']  = df_scaled.groupby('unit')[f'{sensor}'].transform(lambda x: x.rolling(window=5).std())

    # Checks to see if the folder already exists
    output_folder = "rolling_stats_plots"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # This whole entire loops goes through all 21 sensors
    # Plots the data and the rolling average and rolling standard deviations of the data on a graph
    # And saves the graphs as PNG files into a folder
    for sensor_num in range(1,22):
        print(f"Processing sensor {sensor_num} / 21...")
        sensor = f"s{sensor_num}"
        filepath = os.path.join(output_folder, f"sensor_{sensor_num}.png")

        if not os.path.exists(filepath):
            # Creates a blank canvas of size 10 inches wide x 5 inches tall
            plt.figure(figsize=(10,5))

            # Plots the raw sensor data and the standardized sensor data on a graph together
            plt.plot(df_scaled["time"], df_scaled[f"{sensor}"], label=f"Sensor {sensor} (Raw)")
            plt.plot(df_scaled["time"], df_scaled[f"{sensor}_roll_mean"], label=f"Sensor {sensor} (Rolling Avg.)")
            plt.plot(df_scaled["time"], df_scaled[f"{sensor}_roll_std"], label=f"Sensor {sensor} (Rolling Std.)")

            # Organizes the graphs
            plt.xlabel("Time")
            plt.ylabel("Sensor Reading")
            plt.title(f"Sensor {sensor_num} Readings Over Time")
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()

            # Save the graph as a PNG to file
            plt.savefig(filepath)

            # Close the plot (important to prevent memory issues)
            plt.close()
        else:
            print(f"Sensor {sensor_num} already has a saved plot. Skipping.")

    # Sensors with zero or near-zero standard deviation are flat and should be dropped
    sensor_cols = [f"s{i}" for i in range(1,22)] # Creates a list with each sensors name from s1 to s21
    stds = df_scaled[sensor_cols].std() # Finds the standard deviation for each column that corresponds to the value in the sensor_cols list which are the sensors themselves and returns a list
    flat_sensors = stds[stds < 1e-3].index.tolist() # Finds all the sensors with a very low standard deviation (flat sensors), finds the index values of all of them, and converts them to a list
    df_scaled.drop(columns=flat_sensors, inplace=True) # Using the list of flat sensors that were found, they are removed from the dataframe in-place and a new dataframe is not created
    print("Dropped sensors: ", flat_sensors)

    # The following chunk of code adds Delta and Slope features for each sensor which measure that change of the sensor values over time
    # Delta - calculates how much the sensor reading changed from its previous value
    # Slope - calculates the trend of how the sensor reading have changed over a specificed window by fitting a line through the points
    def add_slope_features(df, window=3):
        raw_sensor_cols = [col for col in df.columns if col.startswith('s') and '_roll' not in col]
        for sensor in raw_sensor_cols:
            # Delta (already fine)
            df[f'{sensor}_delta'] = df.groupby('unit')[sensor].diff()

            # Replace slow slope with linear diff of rolling mean
            df[f'{sensor}_slope'] = df.groupby('unit')[sensor].transform(lambda x: x.rolling(window=window).mean().diff())

        return df

    df_scaled = add_slope_features(df_scaled, window=3)

    # The following code chunk is all about selecting which sensors are the most informative for our uses
    # Temporary cleanup for fitting - drop rows with NaNs
    df_scaled.dropna(inplace=True)

    return df_scaled


def phase3(df_selected): # Modeling
    # Separates the features and the target (RUL)
    # x is now a pile of clue (features) the the model will look at - the homework assignment
    x = df_selected.drop(columns=['unit', 'time', 'RUL'])
    y = df_selected['RUL'] # answers to the homework assignment
    units = df_selected['unit']

    # train_test_split() splits that data into two groups: one to teach the computer and one to check if it learned well
    # Split dataset into 80% train and 20% test sets becase test_size is 0.2
    # x_train - clues for training, x_test - answers for training, y_train - answers for training, y_test - answers for testing
    # 80% of the data is given to the model to learn from, the other 20% is used to test the model if it learned well
    x_train, x_test, y_train, y_test, _, units_test = train_test_split( x, y, units, test_size=0.2, random_state=42)
    print(f"Training set size: {x_train.shape[0]} samples")
    print(f"Testing set size: {x_test.shape[0]} samples")

    # Use Random Forest, and measure how well it did using:
    # Mean Absolute Error (MAE) — a score that tells us how “off” the model’s guesses are, on average.

    # Imagine you want to guess how many candies are in a big jar. But it’s tricky, so you ask a bunch of your friends to guess, and then you take an average of all their guesses.
    # This way, you get a better idea than just one guess alone.
    # Random Forest works kind of like that, but with trees instead of friends:
    # Each tree is like one friend who looks at the data and makes a guess.
    # Each tree looks at different parts of the data and asks different questions.
    # Because every tree sees a slightly different view, they make different guesses.
    # When you put all these trees together (a forest!), you take all their guesses and combine them to get a final, better guess.
    # This helps because one tree might be wrong sometimes, but when many trees vote together, the overall guess is more accurate and less likely to be tricked by weird data.
    # So, Random Forest is like a group of smart friends working together to make a good prediction!


    # You are creating an empty brain so that you can use it later
    # You have created 100 'trees' within that brain and you've told it to be accurate every time
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    # Train (fit) the model using the training data
    # You give the model the homework (x) and the answers (y) and ask it to figure out the method to solving the homework based on the answers
    model.fit(x_train, y_train) # Here, we are giving the model the homework assignment and the answers to the homework assignment and tell it to go study
    # Predict RUL using the testing data
    # Telling the model to solve the answers to the homework (x_test) from the 20% we saved earlier, based on the methods that the model figured out earlier from the other 80%
    y_pred = model.predict(x_test)
    # Measure how far off the model's predictions are from the real answers
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    logging.info(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}, R2: {r2:.2f}")

    # Saves this model
    joblib.dump(model, 'rul_model.pkl')

    # feature_importances_ - a NumPy array where each element represents how important a feature was in reducing error across the entire forest
    # index=x.columns - labels each value with the corresponding feature name
    # How importance in computed:
    #   Every time a feature is used to split a node in a tree, it reduces prediction error (impurity)
    #   The model sums these reductions across all trees
    #   The more a feature reduces error across the forest, the higher its importance score.
    importances = pd.Series(model.feature_importances_, index=x.columns).sort_values(ascending=False) # feature_importances_ - how much each feature reduced impurity across the trees
    top_features = importances.head(10).index.tolist()

    return x_train, x_test, y_train, y_test, y_pred, units_test, mae, rmse, r2, top_features, importances


def phase4(x_test, y_test, y_pred, top_features, importances, plot_folder="evaluation_plots"): # Visualization & Evaluation
    os.makedirs(plot_folder, exist_ok=True)

    # Scatter Plot of True RUL vs Predicted RUL
    rul_path = os.path.join(plot_folder, "true_vs_predicted_rul.png")
    if not os.path.exists(rul_path):
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--') # Creates Ideal line
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title("True vs. Predicted Remaining Useful Life")
        plt.tight_layout()
        plt.savefig(rul_path)
        plt.close()

    # Histogram on how off the model was to the actual value
    residuals_path = os.path.join(plot_folder, "residuals_histogram.png")
    if not os.path.exists(residuals_path):
        residuals = y_test - y_pred
        plt.figure(figsize=(8, 5))
        sns.histplot(residuals, kde=True, bins=30, color='purple')
        plt.title("Distribution of Residual Errors")
        plt.xlabel("Error (True RUL - Predicted RUL)")
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(residuals_path)
        plt.close()

    # Barplot of features of the data (sensors?) that were most important in training the model on RUL
    importances_path = os.path.join(plot_folder, "feature_importances.png")
    if not os.path.exists(importances_path):
        importances = pd.Series(index=top_features, data=[importances[feat] for feat in top_features])
        plt.figure(figsize=(10, 5))
        sns.barplot(x=importances.index, y=importances.values, palette="viridis")
        plt.xticks(rotation=45)
        plt.title("Top Feature Importances (by Random Forest)")
        plt.xlabel("Feature")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.savefig(importances_path)
        plt.close()

def engine_error_summary(units_test, y_test, y_pred):
    # Create a dataframe to hold unit, true RUL, pred RUL, and error
    df_errors = pd.DataFrame({'unit': units_test, 'true_RUL': y_test, 'pred_RUL' : y_pred})
    df_errors['abs_error'] = (df_errors['true_RUL'] - df_errors['pred_RUL']).abs()

    # Group by engine, find the average (mean) of all the values in the 'abs_error' columns (find the average error), and make the results look like a table again (rest_index())
    error_summary = df_errors.groupby('unit')['abs_error'].mean().reset_index()
    error_summary = error_summary.sort_values('abs_error')

    return df_errors, error_summary

def select_engines(error_summary):
    # .iloc - lets you grab rows or columns from a dataframe by number, not name
    # df.iloc[0] - gives you the whole first row
    # df.iloc[0]['name'] - gives you just the 'name' in the first row
    # 'unit' - engine
    # This line gives the engine in the very first row of the table
    low_error_engine = error_summary.iloc[0]['unit']
    # This line gives the engine at the halfway point
    medium_error_engine = error_summary.iloc[len(error_summary)//2]['unit']
    # This line gives the engine at the very end
    high_error_engine = error_summary.iloc[-1]['unit']

    selected = [low_error_engine, medium_error_engine, high_error_engine]
    print(f'Selected engines based on error: {selected}')

    return selected

def plot_rul_timelines(df_selected, df_errors, selected_engines, plot_folder="timeline_plots"):
    os.makedirs(plot_folder, exist_ok=True)

    # selected_engines is a list of 3 engines with low error, medium error, and high error
    for engine in selected_engines:
        plot_path = os.path.join(plot_folder, f"engine_{int(engine)}_timeline.png")
        if not os.path.exists(plot_path):
                
            # Make a copy of all the rows that correspond to the current engine in the loop from the error dataframe
            engine_data = df_errors[df_errors['unit'] == engine].copy()

            # Get the row numbers of the copy dataframe
            indices = engine_data.index
            # Go back to the original, cleaned dataframe and get the 'time' values of those corresponding row
            times = df_selected.loc[indices, 'time']

            plt.figure(figsize=(10,5))
            plt.plot(times, engine_data['true_RUL'], label='True RUL')
            plt.plot(times, engine_data['pred_RUL'], label='Predicted RUL')
            plt.title(f'Engine {engine} - True vs Predicted RUL over Time')
            plt.xlabel('Time (cycles)')
            plt.ylabel('Remaining Useful Life (RUL)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(plot_path)
            plt.close()

class PDFWithPageNumbers(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        page_number = f"Page {self.page_no()}"
        self.cell(0, 10, page_number, 0, 0, "C")

def phase5(mae, rmse, r2, error_summary, top_features, plot_folders=["rolling_stats_plots", "evaluation_plots", "timeline_plots"]):
    pdf_path = "engine_rul_report.pdf"
    if os.path.exists(pdf_path):
        print("PDF report already exists. Skipping generation.")
        return

    pdf = PDFWithPageNumbers()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="Engine RUL Prediction Report", ln=True, align='C')

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Arial", "I", 10)
    pdf.cell(0, 10, f"Generated on: {timestamp}", ln=True)
    pdf.ln(10)

    # Key Statistics
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 10, f"Mean Absolute Error (MAE): {mae:.2f}", ln=True)
    pdf.ln(5)
    pdf.cell(0, 10, f"Root Mean Squared Error (RMSE): {rmse:.2f}", ln=True)
    pdf.cell(0, 10, f"R² Score: {r2:.2f}", ln=True)
    pdf.ln(10)
    pdf.cell(0, 10, f"Lowest Error Engine: {int(error_summary.iloc[0]['unit'])} ({error_summary.iloc[0]['abs_error']:.2f})", ln=True)
    pdf.cell(0, 10, f"Highest Error Engine: {int(error_summary.iloc[-1]['unit'])} ({error_summary.iloc[-1]['abs_error']:.2f})", ln=True)
    pdf.ln(10)

    # Top Features
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Top Important Features:", ln=True)
    pdf.set_font("Arial", size=12)
    for feat in top_features:
        pdf.cell(0, 10, f"- {feat}", ln=True)
    pdf.ln(10)

    # Add images from all folders
    for folder in plot_folders:
        if os.path.exists(folder):
            for img_file in sorted(os.listdir(folder)):
                if img_file.endswith(".png"):
                    pdf.add_page()
                    pdf.image(os.path.join(folder, img_file), w=180)

    pdf.output(pdf_path)
    print("PDF report saved as 'engine_rul_report.pdf'")


if __name__ == "__main__":
    df = phase1()
    df = phase2(df)
    x_train, x_test, y_train, y_test, y_pred, units_test, mae, rmse, r2, top_features, importances = phase3(df)

    # Calculate error summaries per engine
    df_errors, error_summary = engine_error_summary(units_test, y_test, y_pred)

    # Select engines
    selected_engines = select_engines(error_summary)

    # Visualizations
    plot_rul_timelines(df, df_errors, selected_engines, plot_folder="timeline_plots")
    phase4(x_test, y_test, y_pred, top_features, importances, plot_folder="evaluation_plots")

    # Generate Report
    phase5(mae, rmse, r2, error_summary, top_features)