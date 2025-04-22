import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder where you want to save the plots
output_folder = "plots"

# Check if the folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def parse_file(filename, inverty=False):
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
        if inverty:
            return np.array(Xs), -np.array(Ys), np.array(Statics)
        else:
            return np.array(Xs), np.array(Ys), np.array(Statics)
    else:
        raise Exception("No data")


def plot_cp_slices_x(x, y, pressure, q, title="Pressure coefficients"):
    '''This does not work :('''
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 2)
    x_filtered = x[mask]
    y_filtered = y[mask]
    pressure_filtered = pressure[mask]

    cps = (1 / q) * np.array(pressure_filtered)
    unique_x = np.unique(x_filtered)

    for x_value in unique_x:
        mask_x = (x_filtered == x_value)
        y_at_x = y_filtered[mask_x]
        cp_at_x = cps[mask_x]

        # Normalize x values for each span (y-value)
        y_min, y_max = np.min(y_at_x), np.max(y_at_x)
        y_at_x_normalized = (y_at_x - y_min) / (y_max - y_min)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_at_x_normalized, cp_at_x, marker='o', s=5)  # Scatter plot
        # plt.plot(x_at_y, cp_at_y, 'bo-') #line plot
        plt.xlabel("Normalized X Position")
        plt.ylabel("cp")
        plt.title(f"{title} at X = {x_value}")
        plt.gca().invert_yaxis()  # Invert y-axis
        plt.grid(True)
        plt.show()
        # Save the plot to the specified folder
        plot_filename = os.path.join(output_folder, f"{title.replace(' ', '_')}_X_{x_value}.png")
        plt.savefig(plot_filename)
        plt.close()

def plot_cp_slices(x, y, pressure, q, title="Pressure coefficients"):
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 2)
    x_filtered = x[mask]
    y_filtered = y[mask]
    pressure_filtered = pressure[mask]

    cps = (1 / q) * np.array(pressure_filtered)
    unique_y = np.unique(y_filtered)

    for y_value in unique_y:
        mask_y = (y_filtered == y_value)
        x_at_y = x_filtered[mask_y]
        cp_at_y = cps[mask_y]

        # Normalize x values for each span (y-value)
        x_min, x_max = np.min(x_at_y), np.max(x_at_y)
        x_at_y_normalized = (x_at_y - x_min) / (x_max - x_min)

        plt.figure(figsize=(8, 6))
        plt.scatter(x_at_y_normalized, cp_at_y, marker='o', s=5)  # Scatter plot
        # plt.plot(x_at_y, cp_at_y, 'bo-') #line plot
        plt.xlabel("Normalized X Position")
        plt.ylabel("cp")
        plt.title(f"{title} at Y = {y_value}")
        plt.gca().invert_yaxis()  # Invert y-axis
        plt.grid(True)
        # Save the plot to the specified folder
        plot_filename = os.path.join(output_folder, f"{title.replace(' ', '_')}_Y_{y_value}.png")
        plt.savefig(plot_filename)
        plt.close()

def compute_lift_coefficient(x, y, pressure, q, S):
    """
    Compute the lift coefficient by integrating the pressure distribution separately
    for positive and negative pressures before summing their contributions.
    """
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 2)
    x_filtered = x[mask]
    y_filtered = y[mask]
    pressure_filtered = pressure[mask]

    unique_y = np.unique(y_filtered)
    local_lift = []

    for y_value in unique_y:
        mask_y = (y_filtered == y_value)
        x_at_y = x_filtered[mask_y]
        pressures_at_y = pressure_filtered[mask_y]

        sorted_indices = np.argsort(x_at_y)
        x_sorted = x_at_y[sorted_indices]
        pressures_sorted = pressures_at_y[sorted_indices]

        # Separate positive and negative pressures for integration
        pos_mask = pressures_sorted >= 0
        neg_mask = pressures_sorted < 0

        lift_pos = np.trapezoid(pressures_sorted[pos_mask], x_sorted[pos_mask]) if np.any(pos_mask) else 0.0
        lift_neg = np.trapezoid(pressures_sorted[neg_mask], x_sorted[neg_mask]) if np.any(neg_mask) else 0.0

        lift_at_y = -(lift_pos + lift_neg)  # Negative sign for pressure convention
        local_lift.append(lift_at_y)

    sorted_y_indices = np.argsort(unique_y)
    unique_y_sorted = unique_y[sorted_y_indices]
    local_lift_sorted = np.array(local_lift)[sorted_y_indices]

    L_total = np.trapezoid(local_lift_sorted, unique_y_sorted)
    Cl = L_total / (q * S)

    return Cl, L_total


def plot_lift_distribution(x, y, pressure):
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 2)
    x_filtered = x[mask]
    y_filtered = y[mask]
    pressure_filtered = pressure[mask]

    unique_y = np.unique(y_filtered)
    lift_distribution = []

    for y_value in unique_y:
        mask_y = (y_filtered == y_value)
        x_at_y = x_filtered[mask_y]
        pressures_at_y = pressure_filtered[mask_y]

        sorted_indices = np.argsort(x_at_y)
        x_sorted = x_at_y[sorted_indices]
        pressures_sorted = pressures_at_y[sorted_indices]

        pos_mask = pressures_sorted >= 0
        neg_mask = pressures_sorted < 0

        lift_pos = np.trapezoid(pressures_sorted[pos_mask], x_sorted[pos_mask]) if np.any(pos_mask) else 0.0
        lift_neg = np.trapezoid(pressures_sorted[neg_mask], x_sorted[neg_mask]) if np.any(neg_mask) else 0.0
        lift_at_y = -(lift_pos + lift_neg)
        x_min, x_max = np.min(x_at_y), np.max(x_at_y)
        print("Chord: ", (x_max - x_min))
        # lift_at_y = lift_at_y/(x_max-x_min) # normalize by the chord
        lift_distribution.append(lift_at_y)

    plt.figure(figsize=(8, 6))
    plt.plot(unique_y, lift_distribution, marker='o', color='b', label="Lift Distribution")
    plt.axvline(x=0, color='r', linestyle='--', label="Centerline")
    plt.axvline(x=0.94, color='g', linestyle='--', label="Right kink")
    plt.axvline(x=-0.94, color='g', linestyle='--', label="Left kink")
    plt.axvline(x=0.579, color='b', linestyle='dotted', label="Right TE kink")
    plt.axvline(x=-0.579, color='b', linestyle='dotted', label="Left TE kink")
    plt.xlabel("Spanwise Position (Y Coordinate)")
    plt.ylabel("Lift (Numerically Integrated)")
    plt.title("Lift Distribution Along the Span")
    plt.grid(True)
    plt.legend()
    plt.show()

def plot_all_lift_distributions(datasets, titles):
    """
    Plot lift distributions for multiple datasets in a single figure.
    Each dataset is a tuple (x, y, pressure).
    """
    plt.figure(figsize=(10, 6))

    for (x, y, pressure), label in zip(datasets, titles):
        mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 2)
        x_filtered = x[mask]
        y_filtered = y[mask]
        pressure_filtered = pressure[mask]

        unique_y = np.unique(y_filtered)
        lift_distribution = []

        for y_value in unique_y:
            mask_y = (y_filtered == y_value)
            x_at_y = x_filtered[mask_y]
            pressures_at_y = pressure_filtered[mask_y]

            sorted_indices = np.argsort(x_at_y)
            x_sorted = x_at_y[sorted_indices]
            pressures_sorted = pressures_at_y[sorted_indices]

            pos_mask = pressures_sorted >= 0
            neg_mask = pressures_sorted < 0

            lift_pos = np.trapezoid(pressures_sorted[pos_mask], x_sorted[pos_mask]) if np.any(pos_mask) else 0.0
            lift_neg = np.trapezoid(pressures_sorted[neg_mask], x_sorted[neg_mask]) if np.any(neg_mask) else 0.0
            lift_at_y = -(lift_pos + lift_neg)
            x_min, x_max = np.min(x_at_y), np.max(x_at_y)
            print("Chord: ", (x_max - x_min))
            # lift_at_y = lift_at_y / (x_max - x_min)  # normalize by the chord
            lift_distribution.append(lift_at_y)

        plt.plot(unique_y, lift_distribution, marker='o', label=label)

    # Add reference lines and formatting
    plt.axvline(x=0, color='r', linestyle='--', label="Centerline")
    plt.axvline(x=0.94, color='g', linestyle='--', label="Right kink")
    plt.axvline(x=-0.94, color='g', linestyle='--', label="Left kink")
    plt.axvline(x=0.579, color='b', linestyle='dotted', label="Right TE kink")
    plt.axvline(x=-0.579, color='b', linestyle='dotted', label="Left TE kink")
    plt.xlabel("Spanwise Position (Y Coordinate)")
    plt.ylabel("Lift (Numerically Integrated)")
    plt.title("Lift Distributions for All Cases")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


files = ["00-00.dat", "00-05.dat", "00-10.dat", "00-15.dat", "00-20.dat", "10-00.dat", "10-05.dat", "10-10.dat", "10-15.dat"]

cls = []
total_lifts = []
left_lifts = []
right_lifts = []
alphas = [0,0,0,0,0,10,10,10,10,10]
betas = [0,5,10,15,20,0,5,10,15,20]
index = 0
for filename in files:

    # Parse the file
    x, y, pressure = parse_file(filename)

    # Filter out points where abs(y) > 1.5 and pressure == 0
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 2)
    x_filtered = x[mask]
    y_filtered = y[mask]
    pressure_filtered = pressure[mask]
    title = "Pressures at alpha = " + str(alphas[index]) + " beta = " + str(betas[index])
    # plot_cp_slices(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, title=title)
    # plot_cp_slices_x(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, title=title)

    plot_lift_distribution(x_filtered, y_filtered, pressure_filtered)
    cl, lift = compute_lift_coefficient(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, 1.841)
    total_lifts.append(lift)
    cls.append(cl)
    leftmask = ((y) <= 0)
    x_left = x[leftmask]
    y_left = y[leftmask]
    pressure_left = pressure[leftmask]
    cl, lift = compute_lift_coefficient(x_left, y_left, pressure_left, 0.5*1.176655*34.70916*34.70916, 1.841)
    left_lifts.append(lift)

    rightmask = ((y) >= 0)
    x_right = x[rightmask]
    y_right = y[rightmask]
    pressure_right = pressure[rightmask]
    cl, lift = compute_lift_coefficient(x_right, y_right, pressure_right, 0.5 * 1.176655 * 34.70916 * 34.70916,
                                        1.841)
    right_lifts.append(lift)
    index+=1
# Parse the file
x, y, pressure = parse_file("10-20.dat", inverty=True)
# Filter out points where abs(y) > 1.5 and pressure == 0
mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 2)
x_filtered = x[mask]
y_filtered = y[mask]
pressure_filtered = pressure[mask]
title = "Pressures at alpha = " + str(alphas[index]) + " beta = " + str(betas[index])
# plot_cp_slices(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, title=title)
# plot_cp_slices_x(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, title=title)

plot_lift_distribution(x_filtered, y_filtered, pressure_filtered)
cl, lift = compute_lift_coefficient(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, 1.841)
cls.append(cl)
total_lifts.append(lift)
leftmask = ((y) <= 0)
x_left = x[leftmask]
y_left = y[leftmask]
pressure_left = pressure[leftmask]
cl, lift = compute_lift_coefficient(x_left, y_left, pressure_left, 0.5*1.176655*34.70916*34.70916, 1.841)
left_lifts.append(lift)
rightmask = ((y) >= 0)
x_right = x[rightmask]
y_right = y[rightmask]
pressure_right = pressure[rightmask]
cl, lift = compute_lift_coefficient(x_right, y_right, pressure_right, 0.5 * 1.176655 * 34.70916 * 34.70916,
                                    1.841)
right_lifts.append(lift)

alphas = np.array(alphas)
total_lifts = np.array(total_lifts)
left_lifts = np.array(left_lifts)
right_lifts = np.array(right_lifts)

# Separate data by alpha values
betas_alpha_0 = betas[:5]
betas_alpha_10 = betas[5:]

left_lift_alpha_0 = left_lifts[:5]
right_lift_alpha_0 = right_lifts[:5]
total_lifts_alpha_0 = total_lifts[:5]

left_lift_alpha_10 = left_lifts[5:]
right_lift_alpha_10 = right_lifts[5:]
total_lifts_alpha_10 = total_lifts[5:]

datasets = []
titles = []

for i, filename in enumerate(files[5:] + ["10-20.dat"]):
    x, y, pressure = parse_file(filename, inverty=(filename == "10-20.dat"))
    datasets.append((x, y, pressure))
    titles.append(f"α={alphas[i+5]}°, β={betas[i+5]}°")

plot_all_lift_distributions(datasets, titles)

datasets = []
titles = []

for i, filename in enumerate(files[:5]):
    x, y, pressure = parse_file(filename, inverty=(filename == "10-20.dat"))
    datasets.append((x, y, pressure))
    titles.append(f"α={alphas[i]}°, β={betas[i]}°")

plot_all_lift_distributions(datasets, titles)


# Plot lift of both wings vs. sideslip angle for alpha = 0
plt.figure(figsize=(8, 6))
plt.plot(betas_alpha_0, left_lift_alpha_0, 'bo-', label="Left Wing (\u03B1=0°)")
plt.plot(betas_alpha_0, right_lift_alpha_0, 'ro-', label="Right Wing (\u03B1=0°)")
plt.plot(betas_alpha_0, total_lifts_alpha_0, 'go-', label="Total Wing (\u03B1=0°)")
plt.xlabel("Sideslip Angle (β°)")
plt.ylabel("Lift Force (N)")
plt.title("Lift vs. Sideslip Angle (α = 0°)")
plt.grid(True)
plt.legend()
plt.show()

# Plot lift of both wings vs. sideslip angle for alpha = 10
plt.figure(figsize=(8, 6))
plt.plot(betas_alpha_10, left_lift_alpha_10, 'bo-', label="Left Wing (\u03B1=10°)")
plt.plot(betas_alpha_10, right_lift_alpha_10, 'ro-', label="Right Wing (\u03B1=10°)")
plt.plot(betas_alpha_10, total_lifts_alpha_10, 'go-', label="Total Wing (\u03B1=10°)")
plt.xlabel("Sideslip Angle (β°)")
plt.ylabel("Lift Force (N)")
plt.title("Lift vs. Sideslip Angle (α = 10°)")
plt.grid(True)
plt.legend()
plt.show()

total_cls_alpha_0 = cls[:5]
total_cls_alpha_10 = cls[5:]

# Plot cl of both wings vs. sideslip angle for alpha = 0
plt.figure(figsize=(8, 6))
plt.plot(betas_alpha_0, total_cls_alpha_0, 'ro-', label="Alpha 0 Cl (\u03B1=0°)")
plt.plot(betas_alpha_0, total_cls_alpha_10, 'bo-', label="Alpha 10 Cl (\u03B1=0°)")
plt.xlabel("Sideslip Angle (β°)")
plt.ylabel("Lift Coefficient")
plt.title("Cl vs. Sideslip Angle")
plt.grid(True)
plt.legend()
plt.show()