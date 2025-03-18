import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

def parse_file(filename):
    with open(filename) as f:
        lines = f.readlines()
    reading = False
    Xs = []
    Ys = []
    Statics = []
    numbers = []
    reading_data = None
    limit = None
    Nodes = None
    for line in lines:
        line = line.strip()
        if line.startswith("ZONE T="):
            reading = False
            readingy = False
            Y = ""
            for char in line:
                if char == "Y":
                    readingy = True
                    continue
                elif readingy:
                    if char == '"':
                        Y = float(Y)
                        readingy = False
                        break
                    elif char != '=':
                        Y += char
        elif line.startswith("Nodes="):
            reading = False
            readingnodes = False
            Nodes = ""
            for char in line:
                if char == "=":
                    readingnodes = True
                    continue
                elif readingnodes:
                    if char == ',':
                        Nodes = float(Nodes)
                        readingnodes = False
                        break
                    elif char != '=':
                        Nodes += char
        elif line.startswith("DT"):
            reading = True
            reading_data = 'x'
            numbers = []
            limit = Nodes
            print("Limit: ", limit)
            continue
        if reading:
            if reading_data == 'x':
                for num in line.split():
                    if len(numbers) == limit:
                        try:
                            numbers.append(float(num))  # Convert to float for numerical consistency
                        except ValueError:
                            pass  # Skip non-numeric values
                        Xs.extend(numbers)
                        reading_data = 'y'
                        numbers = []
                        limit = Nodes
                        break
                    try:
                        numbers.append(float(num))  # Convert to float for numerical consistency
                    except ValueError:
                        pass  # Skip non-numeric values
            if reading_data == 'y':
                for num in line.split():
                    if len(numbers) == limit:
                        try:
                            numbers.append(float(num))  # Convert to float for numerical consistency
                        except ValueError:
                            pass  # Skip non-numeric values
                        Ys.extend(numbers)
                        reading_data = 'p'
                        numbers = []
                        limit = Nodes
                        break
                    try:
                        numbers.append(float(num))  # Convert to float for numerical consistency
                    except ValueError:
                        pass  # Skip non-numeric values
            if reading_data == 'p':
                for num in line.split():
                    if len(numbers) == limit:
                        try:
                            numbers.append(float(num))  # Convert to float for numerical consistency
                        except ValueError:
                            pass  # Skip non-numeric values
                        Statics.extend(numbers)
                        reading = False
                        break
                    try:
                        numbers.append(float(num))  # Convert to float for numerical consistency
                    except ValueError:
                        pass  # Skip non-numeric values
    if Xs is not None and Ys is not None and Statics is not None:
        print(len(Xs), len(Ys), len(Statics))
        return np.array(Xs), np.array(Ys), np.array(Statics)
    else:
        raise Exception("No data")

files = ["00-00.dat", "00-05.dat", "00-10.dat", "00-15.dat", "00-20.dat", "10-00.dat", "10-05.dat", "10-10.dat", "10-15.dat", "10-20.dat"]

def plot_lift_distribution(x, y, pressure):
    # Filter out points where pressure == 0 and abs(y) <= 1.5
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 5)
    x_filtered = x[mask]
    y_filtered = y[mask]
    pressure_filtered = pressure[mask]

    # Group by spanwise locations (y)
    unique_y = np.unique(y_filtered)  # Get unique y-coordinates
    lift_distribution = []

    # Calculate the lift distribution and scale by chord length
    for y_value in unique_y:
        mask_y = (y_filtered == y_value)
        pressures_at_y = pressure_filtered[mask_y]  # Pressure values at this y position
        x_at_y = x_filtered[mask_y]  # x-values at this y position

        # Sum the pressures for each spanwise location
        total_pressure = np.sum(pressures_at_y)

        # Calculate the chord as the difference between max and min x at this y value
        chord_at_y = np.max(x_at_y) - np.min(x_at_y)

        # Scale the summed pressures by the chord length at this y position
        scaled_pressure = total_pressure * chord_at_y

        lift_distribution.append(scaled_pressure)

    # Plot the lift distribution along the span
    plt.figure(figsize=(8, 6))
    plt.plot(unique_y, lift_distribution, marker='o', color='b', label="Lift Distribution")
    plt.xlabel("Spanwise Position (Y Coordinate)")
    plt.ylabel("Lift (Scaled by Chord Length)")
    plt.title("Lift Distribution Along the Span")
    plt.grid(True)
    plt.legend()
    plt.show()

for filename in files:

    # Parse the file
    x, y, pressure = parse_file(filename)

    # Filter out points where abs(y) > 1.5 and pressure == 0
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 5)
    x_filtered = x[mask]
    y_filtered = y[mask]
    pressure_filtered = pressure[mask]

    plot_lift_distribution(x_filtered, y_filtered, pressure_filtered)
    # # Create a 3D plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')
    #
    # # Plot the filtered data as tiny dots in 3D
    # sc = ax.scatter(x_filtered, y_filtered, pressure_filtered, c=pressure_filtered, cmap='viridis', s=5, edgecolor='k')
    #
    # # Add color bar
    # plt.colorbar(sc, label="Static Pressure")
    #
    # # Labels and title
    # ax.set_xlabel("X Coordinate")
    # ax.set_ylabel("Y Coordinate")
    # ax.set_zlabel("Static Pressure")
    # ax.set_title("3D Static Pressure Distribution (Filtered")
    #
    # # Show the plot
    # plt.show()