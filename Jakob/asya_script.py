import numpy as np
import pandas as pd
import wfdb
from IPython.display import display
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats
from scipy.fft import fft
from scipy.signal import butter, filtfilt


record = wfdb.rdrecord('term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_t001') 
wfdb.plot_wfdb(record=record) 
display(record.__dict__)

def get_records(max, prefix):   
    return_me = []
    for x in range(1, max+1):
        add_me = ""
        if (x < 10):
            add_me = "0" + str(x)
        else:
            add_me = str(x)
        return_me.append(wfdb.rdrecord('term-preterm-ehg-dataset-with-tocogram-1.0.0/tpehgt_' + prefix +'0' + add_me) )
    return return_me

control_records = get_records(5, 'n')
term_records = get_records(13, 't')
preterm_records = get_records(13, 'p')

# for term in term_records:
#     data = 


def get_features(record):
    data = [row[7] for row in record.__dict__['p_signal']]
    mean = np.mean(data)
    median = np.median(data)
    std_dev = np.std(data)
    min_val = np.min(data)
    max_val = np.max(data)
    range_val = max_val - min_val
    skewness = stats.skew(data)
    kurtosis = stats.kurtosis(data)
    fourier_transform = fft(data)
    # fft_magnitude = np.abs(fourier_transform)[0]
    # fft_features = fft_magnitude[1:5]

    return [mean, median, std_dev, min_val, max_val, range_val, skewness, kurtosis] #, fft_magnitude,  fft_features]

X = []
Y = []

for record in term_records:
    X.append(get_features(record))
    Y.append(1)
for record in preterm_records:
    X.append(get_features(record))
    Y.append(0)
    


X[0]

X = np.array(X)
y = np.array(Y)
# Normalize the input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = keras.Sequential([
    keras.layers.Dense(16, activation='relu'),  # 16 neurons in hidden layer, 3 input features
    keras.layers.Dense(8, activation='relu'),  # 8 neurons in hidden layer
    keras.layers.Dense(1, activation='sigmoid')  # 1 neuron for binary classification (sigmoid)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")


def butterworth_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalize the cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Design filter
    y = filtfilt(b, a, data)  # Apply filter with zero-phase filtering
    return y

# Parameters for the Butterworth filter
cutoff_frequency = 2.0  # Desired cutoff frequency (Hz)
sampling_rate = 50.0  # Sampling rate of your data (samples per second)
filter_order = 4  # Order of the filter

# Apply the Butterworth filter to the 'data' column in the DataFrame
df['filtered_data'] = butterworth_filter(df['data'], cutoff_frequency, sampling_rate, filter_order)

# Display the original and filtered data
print(df.head())

# Optionally, you can plot the original and filtered signals
import matplotlib.pyplot as plt

plt.plot(df['time'], df['data'], label='Original data')
plt.plot(df['time'], df['filtered_data'], label='Filtered data', color='red')
plt.legend()
plt.show()