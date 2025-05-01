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
    Zs = []
    Statics = []
    numbers = []
    reading_data = None
    limit = None
    Nodes = None
    for line in lines:
        line = line.strip()
        if line.startswith("ZONE T="):
            reading = False
            readingz = False
            Z = ""
            skipped_firstZ = False
            for char in line:
                if char == "Z":
                    if skipped_firstZ:
                        readingz = True
                    else:
                        skipped_firstZ = True
                    continue
                elif readingz:
                    if char == ' ':
                        Z = float(Z)
                        readingz = False
                        break
                    elif char != '=':
                        Z += char
        elif line.startswith("I="):
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
            # print("Limit: ", limit)
            continue
        if reading:
            if reading_data == 'x':
                for num in line.split():
                    try:
                        numbers.append(float(num))  # Convert to float for numerical consistency
                    except ValueError:
                        pass  # Skip non-numeric values
                    if len(numbers) == limit:
                        Xs.extend(numbers)
                        reading_data = 'y'
                        numbers = []
                        limit = Nodes
                        break
            if reading_data == 'y':
                for num in line.split():
                    try:
                        numbers.append(float(num))  # Convert to float for numerical consistency
                    except ValueError:
                        pass  # Skip non-numeric values
                    if len(numbers) == limit:
                        Ys.extend(numbers)
                        reading_data = 'z'
                        numbers = []
                        limit = Nodes
                        break
            if reading_data == 'z':
                for num in line.split():
                    try:
                        numbers.append(float(num))  # Convert to float for numerical consistency
                    except ValueError:
                        pass  # Skip non-numeric values
                    if len(numbers) == limit:
                        Zs.extend(numbers)
                        reading_data = 'p'
                        numbers = []
                        limit = Nodes
                        break
            if reading_data == 'p':
                for num in line.split():

                    try:
                        numbers.append(float(num))# Convert to float for numerical consistency
                    except ValueError:
                        pass  # Skip non-numeric values

                    if len(numbers) == limit:
                        Statics.extend(numbers)
                        reading = False
                        break
    if Xs is not None and Ys is not None and Statics is not None:
        print(len(Xs), len(Ys), len(Zs), len(Statics))
        if inverty:
            print(Xs[0:3], Ys[0:3], Zs[0:3], Statics[0:3])
            return np.array(Xs), -np.array(Ys), np.array(Zs),np.array(Statics)
        else:
            print(Xs[0:3], Ys[0:3], Zs[0:3], Statics[0:3])
            return np.array(Xs), np.array(Ys), np.array(Zs),np.array(Statics)
    else:
        raise Exception("No data")


def compute_lift_coefficient(x, y, z, pressure, q, S):
    """
    Compute the lift coefficient by integrating the pressure distribution separately
    for positive and negative pressures before summing their contributions.
    """
    mask = (np.abs(y) <= 1.6) & (np.abs(pressure) > 0.5)
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    pressure_filtered = pressure[mask]

    unique_z = np.unique(z_filtered)
    # print(unique_y)
    local_lift = []

    for z_value in unique_z:
        mask_z = (y_filtered == z_value)
        x_at_z = x_filtered[mask_z]
        y_at_z = y_filtered[mask_z]
        pressures_at_z = pressure_filtered[mask_z]

        sorted_indices = np.argsort(x_at_z)
        x_sorted = x_at_z[sorted_indices]
        pressures_sorted = pressures_at_z[sorted_indices]
        y_sorted = y_at_z[sorted_indices]

        # Assume x_sorted and z_sorted are already defined and sorted by x
        x_sorted = np.asarray(x_sorted)
        y_sorted = np.asarray(y_sorted)

        # Define midpoint at the point of maximum Z
        mid_index = np.argmax(y_sorted)
        x_mid = x_sorted[mid_index]

        # Leading edge segment
        x_lead = x_sorted[:(mid_index + 1)]
        y_lead = y_sorted[:(mid_index + 1)]
        m1 = (y_lead[-1] - y_lead[0]) / (x_lead[-1] - x_lead[0])
        line1 = y_lead[0] + m1 * (x_lead - x_lead[0])

        # Trailing edge segment
        x_trail = x_sorted[mid_index:]
        y_trail = y_sorted[mid_index:]
        m2 = (y_trail[-1] - y_trail[0]) / (x_trail[-1] - x_trail[0])
        line2 = y_trail[0] + m2 * (x_trail - x_trail[0])

        # Construct the full separating line
        boundary = np.empty_like(y_sorted)
        boundary[:(mid_index + 1)] = line1
        boundary[mid_index:] = line2

        # Create masks
        pos_mask = y_sorted >= boundary
        neg_mask = y_sorted < boundary

        lift_pos = np.trapezoid(pressures_sorted[pos_mask], x_sorted[pos_mask]) if np.any(pos_mask) else 0.0
        lift_neg = np.trapezoid(pressures_sorted[neg_mask], x_sorted[neg_mask]) if np.any(neg_mask) else 0.0

        lift_at_z = -(lift_pos + lift_neg)  # Negative sign for pressure convention
        local_lift.append(lift_at_z)

    sorted_z_indices = np.argsort(unique_z)
    unique_z_sorted = unique_z[sorted_z_indices]
    local_lift_sorted = np.array(local_lift)[sorted_z_indices]

    L_total = np.trapezoid(local_lift_sorted, unique_z_sorted)
    Cl = L_total / (q * S)

    return Cl, L_total


def plot_lift_distribution(x, y, z, pressure, plot_surf=False, invertlift=False):
    mask = (np.abs(y) <= 1.6) & (np.abs(pressure) > 0.5)
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    pressure_filtered = pressure[mask]

    unique_z = np.unique(z_filtered)
    lift_distribution = []
    valid_z_values = []

    for z_value in unique_z:
        mask_z = (z_filtered == z_value)
        x_at_z = x_filtered[mask_z]
        y_at_z = y_filtered[mask_z]
        pressures_at_z = pressure_filtered[mask_z]

        sorted_indices = np.argsort(x_at_z)
        x_sorted = x_at_z[sorted_indices]
        y_sorted = y_at_z[sorted_indices]
        pressures_sorted = pressures_at_z[sorted_indices]

        # Assume x_sorted and z_sorted are already defined and sorted by x
        x_sorted = np.asarray(x_sorted)
        y_sorted = np.asarray(y_sorted)

        # Define midpoint at the point of maximum Z
        mid_index = np.argmax(y_sorted)
        x_mid = x_sorted[mid_index]

        # Leading edge segment
        x_lead = x_sorted[:(mid_index+1)]
        y_lead = y_sorted[:(mid_index+1)]
        m1 = (y_lead[-1] - y_lead[0]) / (x_lead[-1] - x_lead[0])
        line1 = y_lead[0] + m1 * (x_lead - x_lead[0])

        # Trailing edge segment
        x_trail = x_sorted[mid_index:]
        y_trail = y_sorted[mid_index:]
        m2 = (y_trail[-1] - y_trail[0]) / (x_trail[-1] - x_trail[0])
        line2 = y_trail[0] + m2 * (x_trail - x_trail[0])

        # Construct the full separating line
        boundary = np.empty_like(y_sorted)
        boundary[:(mid_index+1)] = line1
        boundary[mid_index:] = line2

        # Create masks
        pos_mask = y_sorted >= boundary
        neg_mask = y_sorted < boundary

        # Plotting
        if plot_surf:
            plt.plot(x_sorted[pos_mask], y_sorted[pos_mask], label=f'Upper surface at {z_value}', linewidth=2)
            plt.plot(x_sorted[neg_mask], y_sorted[neg_mask], label=f'Lower surface at {z_value}', linewidth=2)

            # Plot the separating lines
            plt.plot(x_lead, line1, 'k--', label='Leading edge split line', alpha=0.7)
            plt.plot(x_trail, line2, 'k--', label='Trailing edge split line', alpha=0.7)

            # Improve appearance
            plt.xlabel('x-coordinate')
            plt.ylabel('y-coordinate')
            plt.title('Airfoil Upper and Lower Surfaces')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()


        lift_pos = np.trapezoid(pressures_sorted[pos_mask], x_sorted[pos_mask]) if np.any(pos_mask) else 0.0
        lift_neg = np.trapezoid(pressures_sorted[neg_mask], x_sorted[neg_mask]) if np.any(neg_mask) else 0.0
        lift_at_z = -(lift_pos + lift_neg)
        if invertlift:
            lift_at_z = 0-lift_at_z
        x_min, x_max = np.min(x_at_z), np.max(x_at_z)
        if (x_max - x_min) < 0.0001:
            continue
        print("Chord: ", (x_max - x_min))
        # lift_at_y = lift_at_y/(x_max-x_min) # normalize by the chord
        valid_z_values.append(z_value)
        lift_distribution.append(lift_at_z)

    plt.figure(figsize=(8, 6))
    plt.plot(valid_z_values, lift_distribution, marker='o', color='b', label="Lift Distribution")
    plt.xlabel("Spanwise Position (Z Coordinate)")
    plt.ylabel("Lift (Numerically Integrated)")
    plt.title("Lift Distribution Along the Span")
    plt.grid(True)
    plt.legend()

    filename_export = filename+'_Lift_Distribution.png'

    plt.savefig(os.path.join(output_folder, filename_export), dpi=300)

    plt.show()

def plot_all_lift_distributions(datasets, titles):
    """
    Plot lift distributions for multiple datasets in a single figure.
    Each dataset is a tuple (x, y, z, pressure).
    """
    plt.figure(figsize=(10, 6))

    for (x, y, z, pressure), label in zip(datasets, titles):
        mask = (np.abs(pressure) > 1)
        x_filtered = x[mask]
        y_filtered = y[mask]
        z_filtered = z[mask]
        pressure_filtered = pressure[mask]

        unique_z = np.unique(z_filtered)
        lift_distribution = []
        valid_z_values = []

        for z_value in unique_z:
            mask_z = (z_filtered == z_value)
            x_at_z = x_filtered[mask_z]
            y_at_z = y_filtered[mask_z]
            pressures_at_z = pressure_filtered[mask_z]

            sorted_indices = np.argsort(x_at_z)
            x_sorted = x_at_z[sorted_indices]
            y_sorted = y_at_z[sorted_indices]
            pressures_sorted = pressures_at_z[sorted_indices]

            # Assume x_sorted and z_sorted are already defined and sorted by x
            x_sorted = np.asarray(x_sorted)
            y_sorted = np.asarray(y_sorted)

            # Define midpoint at the point of maximum Z
            mid_index = np.argmax(y_sorted)
            x_mid = x_sorted[mid_index]

            # Leading edge segment
            x_lead = x_sorted[:(mid_index + 1)]
            y_lead = y_sorted[:(mid_index + 1)]
            m1 = (y_lead[-1] - y_lead[0]) / (x_lead[-1] - x_lead[0])
            line1 = y_lead[0] + m1 * (x_lead - x_lead[0])

            # Trailing edge segment
            x_trail = x_sorted[mid_index:]
            y_trail = y_sorted[mid_index:]
            m2 = (y_trail[-1] - y_trail[0]) / (x_trail[-1] - x_trail[0])
            line2 = y_trail[0] + m2 * (x_trail - x_trail[0])

            # Construct the full separating line
            boundary = np.empty_like(y_sorted)
            boundary[:(mid_index + 1)] = line1
            boundary[mid_index:] = line2

            # Create masks
            pos_mask = y_sorted >= boundary
            neg_mask = y_sorted < boundary


            lift_pos = np.trapezoid(pressures_sorted[pos_mask], x_sorted[pos_mask]) if np.any(pos_mask) else 0.0
            lift_neg = np.trapezoid(pressures_sorted[neg_mask], x_sorted[neg_mask]) if np.any(neg_mask) else 0.0
            lift_at_z = -(lift_pos + lift_neg)
            if label == f"α=10°, β=20°":
                lift_at_z = 0-lift_at_z # Invert the α=10°, β=20° case
            x_min, x_max = np.min(x_at_z), np.max(x_at_z)
            if (x_max - x_min) < 0.0001:
                continue
            print("Chord: ", (x_max - x_min))
            # lift_at_y = lift_at_y/(x_max-x_min) # normalize by the chord
            valid_z_values.append(z_value)
            lift_distribution.append(lift_at_z)

        plt.plot(valid_z_values, lift_distribution, marker='o', label=label)

    # Add reference lines and formatting
    plt.xlabel("Spanwise Position (Z Coordinate)")
    plt.ylabel("Lift (Numerically Integrated)")
    plt.title("Lift Distributions for All Cases")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename_export = titles[0] + '_Lift_Distributions.png'

    plt.savefig(os.path.join(output_folder, filename_export), dpi=300)

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
    x, y, z, pressure = parse_file(filename)

    # Filter out points where abs(y) > 10.5 and pressure == 0
    mask = (np.abs(pressure) > 1)
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    pressure_filtered = pressure[mask]
    title = "Pressures at alpha = " + str(alphas[index]) + " beta = " + str(betas[index])

    plot_lift_distribution(x_filtered, y_filtered, z_filtered,pressure_filtered)


# Parse the file
x, y, z, pressure = parse_file("10-20.dat", inverty=True)
# Filter out points where abs(y) > 10.5 and pressure == 0
mask = (np.abs(pressure) > 1)
x_filtered = x[mask]
y_filtered = y[mask]
z_filtered = z[mask]
pressure_filtered = pressure[mask]
title = "Pressures at alpha = " + str(alphas[index]) + " beta = " + str(betas[index])
# plot_cp_slices(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, title=title)
# plot_cp_slices_x(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, title=title)

plot_lift_distribution(x_filtered, y_filtered, z_filtered, pressure_filtered)


alphas = np.array(alphas)

# Separate data by alpha values
betas_alpha_0 = betas[:5]
betas_alpha_10 = betas[5:]

datasets = []
titles = []

for i, filename in enumerate(files[5:] + ["10-20.dat"]):
    x, y, z, pressure = parse_file(filename, inverty=(filename == "10-20.dat"))
    datasets.append((x, y, z, pressure))
    titles.append(f"α={alphas[i+5]}°, β={betas[i+5]}°")

plot_all_lift_distributions(datasets, titles)

datasets = []
titles = []

for i, filename in enumerate(files[:5]):
    x, y, z, pressure = parse_file(filename, inverty=(filename == "10-20.dat"))
    datasets.append((x, y, z, pressure))
    titles.append(f"α={alphas[i]}°, β={betas[i]}°")

plot_all_lift_distributions(datasets, titles)
