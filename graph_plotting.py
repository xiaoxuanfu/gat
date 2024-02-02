import matplotlib.pyplot as plt
import numpy as np

d_model = [32, 64, 128, 256, 512, 1024]

train_accuracy = [83.06, 94.17, 97.22, 98.06, 99.44, 72.78]
validation_accuracy = [47.80, 57.20, 59.47, 59.07, 61.80, 44.13]
test_accuracy = [49.17, 57.47, 61.00, 61.27, 61.33, 58.47]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(d_model, train_accuracy, marker='o', label='Train Accuracy')
plt.plot(d_model, validation_accuracy, marker='o', label='Validation Accuracy')
plt.plot(d_model, test_accuracy, marker='o', label='Test Accuracy')

# Adding labels and title
plt.xlabel('d_model')
plt.ylabel('Accuracy')
plt.title('Accuracy vs d_model (CiteSeer Dataset, n = 16)')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_vs_dmodel_citeseer.png')

plt.show()
