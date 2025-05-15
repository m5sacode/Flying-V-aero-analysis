import numpy as np
import matplotlib.pyplot as plt
import os

# Define the folder where you want to save the plots
output_folder = "plots"

# y_values = [-1.48, -1.40780488, -1.33560976, -1.26341463, -1.19121951, -1.11902439, -1.04682927, -0.97463415, -0.90243902, -0.8302439, -0.75804878, -0.68585366
# , -0.61365854, -0.54146341, -0.46926829, -0.39707317, -0.32487805, -0.25268293
# , -0.1804878, -0.10829268, -0.03609756, 0.03609756, 0.10829268, 0.1804878
# , 0.25268293, 0.32487805, 0.39707317, 0.46926829, 0.54146341, 0.61365854
# , 0.68585366, 0.75804878, 0.830243, 0.90243902, 0.97463415, 1.04682927
# , 1.11902439, 1.19121951, 1.26341463, 1.33560976, 1.40780488, 1.48]

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
                        reading_data = 'z'
                        numbers = []
                        limit = Nodes
                        break
                    try:
                        numbers.append(float(num))  # Convert to float for numerical consistency
                    except ValueError:
                        pass  # Skip non-numeric values
            if reading_data == 'z':
                for num in line.split():
                    if len(numbers) == limit:
                        try:
                            numbers.append(float(num))  # Convert to float for numerical consistency
                        except ValueError:
                            pass  # Skip non-numeric values
                        Zs.extend(numbers)
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
            return np.array(Xs), -np.array(Ys), np.array(Zs),np.array(Statics)
        else:
            return np.array(Xs), np.array(Ys), np.array(Zs),np.array(Statics)
    else:
        raise Exception("No data")


def compute_lift_coefficient(x, y, z, pressure, q, S):
    """
    Compute the lift coefficient by integrating the pressure distribution separately
    for positive and negative pressures before summing their contributions.
    """
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 10)
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    pressure_filtered = pressure[mask]

    unique_y = np.unique(y_filtered)
    # print(unique_y)
    local_lift = []

    for y_value in unique_y:
        mask_y = (y_filtered == y_value)
        x_at_y = x_filtered[mask_y]
        z_at_y = z_filtered[mask_y]
        pressures_at_y = pressure_filtered[mask_y]

        sorted_indices = np.argsort(x_at_y)
        x_sorted = x_at_y[sorted_indices]
        pressures_sorted = pressures_at_y[sorted_indices]
        z_sorted = z_at_y[sorted_indices]

        # Assume x_sorted and z_sorted are already defined and sorted by x
        x_sorted = np.asarray(x_sorted)
        z_sorted = np.asarray(z_sorted)

        # Define midpoint at the point of maximum Z
        mid_index = np.argmax(z_sorted)
        x_mid = x_sorted[mid_index]

        # Leading edge segment
        x_lead = x_sorted[:(mid_index + 1)]
        z_lead = z_sorted[:(mid_index + 1)]
        m1 = (z_lead[-1] - z_lead[0]) / (x_lead[-1] - x_lead[0])
        line1 = z_lead[0] + m1 * (x_lead - x_lead[0])

        # Trailing edge segment
        x_trail = x_sorted[mid_index:]
        z_trail = z_sorted[mid_index:]
        m2 = (z_trail[-1] - z_trail[0]) / (x_trail[-1] - x_trail[0])
        line2 = z_trail[0] + m2 * (x_trail - x_trail[0])

        # Construct the full separating line
        boundary = np.empty_like(z_sorted)
        boundary[:(mid_index + 1)] = line1
        boundary[mid_index:] = line2

        # Create masks
        pos_mask = z_sorted >= boundary
        neg_mask = z_sorted < boundary

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


def plot_lift_distribution(x, y, z, pressure, plot_surf=False, q=0.5 * 1.176655 * 34.70916 * 34.70916, S=1.841):
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 10)
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    pressure_filtered = pressure[mask]

    unique_y = np.unique(y_filtered)
    lift_distribution = []
    valid_y_values = []

    for y_value in unique_y:
        mask_y = (y_filtered == y_value)
        x_at_y = x_filtered[mask_y]
        z_at_y = z_filtered[mask_y]
        pressures_at_y = pressure_filtered[mask_y]

        sorted_indices = np.argsort(x_at_y)
        x_sorted = x_at_y[sorted_indices]
        z_sorted = z_at_y[sorted_indices]
        pressures_sorted = pressures_at_y[sorted_indices]

        # Assume x_sorted and z_sorted are already defined and sorted by x
        x_sorted = np.asarray(x_sorted)
        z_sorted = np.asarray(z_sorted)

        # Define midpoint at the point of maximum Z
        mid_index = np.argmax(z_sorted)
        x_mid = x_sorted[mid_index]

        # Leading edge segment
        x_lead = x_sorted[:(mid_index+1)]
        z_lead = z_sorted[:(mid_index+1)]
        m1 = (z_lead[-1] - z_lead[0]) / (x_lead[-1] - x_lead[0])
        line1 = z_lead[0] + m1 * (x_lead - x_lead[0])

        # Trailing edge segment
        x_trail = x_sorted[mid_index:]
        z_trail = z_sorted[mid_index:]
        m2 = (z_trail[-1] - z_trail[0]) / (x_trail[-1] - x_trail[0])
        line2 = z_trail[0] + m2 * (x_trail - x_trail[0])

        # Construct the full separating line
        boundary = np.empty_like(z_sorted)
        boundary[:(mid_index+1)] = line1
        boundary[mid_index:] = line2

        # Create masks
        pos_mask = z_sorted >= boundary
        neg_mask = z_sorted < boundary

        # Plotting
        if plot_surf:
            plt.plot(x_sorted[pos_mask], z_sorted[pos_mask], label=f'Upper surface at {y_value}', linewidth=2)
            plt.plot(x_sorted[neg_mask], z_sorted[neg_mask], label=f'Lower surface at {y_value}', linewidth=2)

            # Plot the separating lines
            plt.plot(x_lead, line1, 'k--', label='Leading edge split line', alpha=0.7)
            plt.plot(x_trail, line2, 'k--', label='Trailing edge split line', alpha=0.7)

            # Improve appearance
            plt.xlabel('x-coordinate')
            plt.ylabel('z-coordinate')
            plt.title('Airfoil Upper and Lower Surfaces')
            plt.gca().set_aspect('equal', adjustable='box')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()


        lift_pos = np.trapezoid(pressures_sorted[pos_mask], x_sorted[pos_mask]) if np.any(pos_mask) else 0.0
        lift_neg = np.trapezoid(pressures_sorted[neg_mask], x_sorted[neg_mask]) if np.any(neg_mask) else 0.0
        lift_at_y = -(lift_pos + lift_neg)
        x_min, x_max = np.min(x_at_y), np.max(x_at_y)
        if (x_max - x_min) < 0.0001:
            continue
        print("Chord: ", (x_max - x_min))
        chord = x_max - x_min
        lift_at_y = lift_at_y*(x_max-x_min) # denormalize by the chord
        lift_at_y = lift_at_y/(q*S*chord) # make it a lift coefficient
        valid_y_values.append(y_value)
        lift_distribution.append(lift_at_y)

    plt.figure(figsize=(8, 6))
    plt.plot(valid_y_values, lift_distribution, marker='o', color='b', label="Lift Distribution")
    plt.axvline(x=0, color='r', linestyle='--', label="Centerline")
    plt.axvline(x=0.94, color='g', linestyle='--', label="Right kink")
    plt.axvline(x=-0.94, color='g', linestyle='--', label="Left kink")
    plt.axvline(x=0.579, color='b', linestyle='dotted', label="Right TE kink")
    plt.axvline(x=-0.579, color='b', linestyle='dotted', label="Left TE kink")
    plt.xlabel("Spanwise Position (Y Coordinate)")
    plt.ylabel("Cl * chord (Numerically Integrated)")
    plt.title("Lift Distribution Along the Span")
    plt.grid(True)
    plt.legend()

    filename_export = filename+'_Lift_Distribution.png'

    plt.savefig(os.path.join(output_folder, filename_export), dpi=300)

    plt.show()

def plot_all_lift_distributions(datasets, titles, q=0.5 * 1.176655 * 34.70916 * 34.70916, S=1.841, leyend=False):
    """
    Plot lift distributions for multiple datasets in a single figure.
    Each dataset is a tuple (x, y, z, pressure).
    """
    plt.figure(figsize=(10, 6))

    for (x, y, z, pressure), label in zip(datasets, titles):
        mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 10)
        x_filtered = x[mask]
        y_filtered = y[mask]
        z_filtered = z[mask]
        pressure_filtered = pressure[mask]

        unique_y = np.unique(y_filtered)
        lift_distribution = []
        valid_y_values = []

        for y_value in unique_y:
            mask_y = (y_filtered == y_value)
            x_at_y = x_filtered[mask_y]
            z_at_y = z_filtered[mask_y]
            pressures_at_y = pressure_filtered[mask_y]

            sorted_indices = np.argsort(x_at_y)
            x_sorted = x_at_y[sorted_indices]
            z_sorted = z_at_y[sorted_indices]
            pressures_sorted = pressures_at_y[sorted_indices]

            # Assume x_sorted and z_sorted are already defined and sorted by x
            x_sorted = np.asarray(x_sorted)
            z_sorted = np.asarray(z_sorted)

            # Define midpoint at the point of maximum Z
            mid_index = np.argmax(z_sorted)
            x_mid = x_sorted[mid_index]

            # Leading edge segment
            x_lead = x_sorted[:(mid_index + 1)]
            z_lead = z_sorted[:(mid_index + 1)]
            m1 = (z_lead[-1] - z_lead[0]) / (x_lead[-1] - x_lead[0])
            line1 = z_lead[0] + m1 * (x_lead - x_lead[0])

            # Trailing edge segment
            x_trail = x_sorted[mid_index:]
            z_trail = z_sorted[mid_index:]
            m2 = (z_trail[-1] - z_trail[0]) / (x_trail[-1] - x_trail[0])
            line2 = z_trail[0] + m2 * (x_trail - x_trail[0])

            # Construct the full separating line
            boundary = np.empty_like(z_sorted)
            boundary[:(mid_index + 1)] = line1
            boundary[mid_index:] = line2

            # Create masks
            pos_mask = z_sorted >= boundary
            neg_mask = z_sorted < boundary

            lift_pos = np.trapezoid(pressures_sorted[pos_mask], x_sorted[pos_mask]) if np.any(pos_mask) else 0.0
            lift_neg = np.trapezoid(pressures_sorted[neg_mask], x_sorted[neg_mask]) if np.any(neg_mask) else 0.0
            lift_at_y = -(lift_pos + lift_neg)
            x_min, x_max = np.min(x_at_y), np.max(x_at_y)
            if (x_max - x_min) < 0.0001:
                continue
            print("Chord: ", (x_max - x_min))
            # lift_at_y = lift_at_y/(x_max-x_min) # normalize by the chord
            valid_y_values.append(y_value)
            chord = x_max - x_min
            lift_at_y = lift_at_y*(x_max-x_min) # denormalize by the chord
            lift_at_y = lift_at_y / (q * S * chord)  # make it a lift coefficient
            lift_distribution.append(lift_at_y)

        plt.plot(valid_y_values, lift_distribution, marker='o', label=label)

    # Add reference lines and formatting
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='r', linestyle='--', label="Centerline")
    plt.axvline(x=0.94, color='g', linestyle='--', label="Right kink")
    plt.axvline(x=-0.94, color='g', linestyle='--', label="Left kink")
    plt.axvline(x=0.579, color='b', linestyle='dotted', label="Right TE kink")
    plt.axvline(x=-0.579, color='b', linestyle='dotted', label="Left TE kink")
    plt.xlabel("Spanwise Position (Y Coordinate in meters)")
    plt.ylabel("Cl * chord (Numerically Integrated)")
    # plt.title("Lift Distributions for All Cases")
    plt.grid(True)
    if leyend:
        plt.legend(fontsize=15)
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
    mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 10)
    x_filtered = x[mask]
    y_filtered = y[mask]
    z_filtered = z[mask]
    pressure_filtered = pressure[mask]
    title = "Pressures at alpha = " + str(alphas[index]) + " beta = " + str(betas[index])

    plot_lift_distribution(x_filtered, y_filtered, z_filtered,pressure_filtered)
    cl, lift = compute_lift_coefficient(x_filtered, y_filtered, z_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, 1.841)
    total_lifts.append(lift)
    cls.append(cl)
    leftmask = ((y) <= 0)
    x_left = x[leftmask]
    y_left = y[leftmask]
    z_left = z[leftmask]
    pressure_left = pressure[leftmask]
    cl, lift = compute_lift_coefficient(x_left, y_left, z_left, pressure_left, 0.5*1.176655*34.70916*34.70916, 1.841)
    left_lifts.append(lift)

    rightmask = ((y) >= 0)
    x_right = x[rightmask]
    y_right = y[rightmask]
    z_right = z[rightmask]
    pressure_right = pressure[rightmask]
    cl, lift = compute_lift_coefficient(x_right, y_right, z_right, pressure_right, 0.5 * 1.176655 * 34.70916 * 34.70916,
                                        1.841)
    right_lifts.append(lift)
    index+=1
# Parse the file
x, y, z, pressure = parse_file("10-20.dat", inverty=True)
# Filter out points where abs(y) > 10.5 and pressure == 0
mask = (np.abs(y) <= 1.5) & (np.abs(pressure) > 10)
x_filtered = x[mask]
y_filtered = y[mask]
z_filtered = z[mask]
pressure_filtered = pressure[mask]
title = "Pressures at alpha = " + str(alphas[index]) + " beta = " + str(betas[index])
# plot_cp_slices(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, title=title)
# plot_cp_slices_x(x_filtered, y_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, title=title)

plot_lift_distribution(x_filtered, y_filtered, z_filtered, pressure_filtered)
cl, lift = compute_lift_coefficient(x_filtered, y_filtered, z_filtered, pressure_filtered, 0.5*1.176655*34.70916*34.70916, 1.841)
cls.append(cl)
total_lifts.append(lift)
leftmask = ((y) <= 0)
x_left = x[leftmask]
y_left = y[leftmask]
z_left = z[leftmask]
pressure_left = pressure[leftmask]
cl, lift = compute_lift_coefficient(x_left, y_left, z_left, pressure_left, 0.5*1.176655*34.70916*34.70916, 1.841)
left_lifts.append(lift)
rightmask = ((y) >= 0)
x_right = x[rightmask]
y_right = y[rightmask]
z_right = z[rightmask]
pressure_right = pressure[rightmask]
cl, lift = compute_lift_coefficient(x_right, y_right, z_right, pressure_right, 0.5 * 1.176655 * 34.70916 * 34.70916,
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
    x, y, z, pressure = parse_file(filename, inverty=(filename == "10-20.dat"))
    datasets.append((x, y, z, pressure))
    titles.append(f"β={betas[i+5]}°")

plot_all_lift_distributions(datasets, titles)

datasets = []
titles = []

for i, filename in enumerate(files[:5]):
    x, y, z, pressure = parse_file(filename, inverty=(filename == "10-20.dat"))
    datasets.append((x, y, z, pressure))
    titles.append(f"β={betas[i]}°")

plot_all_lift_distributions(datasets, titles, leyend=True)


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