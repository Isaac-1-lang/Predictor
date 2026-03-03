"""
Projectile Range-Angle Predictor
--------------------------------

This script uses simple polynomial regression to:
- Model **range as a function of launch angle** (angle → range)
- Model **angle as a function of range** (range → angle)
- Predict the range for a launch angle of 50 degrees
- Let the user enter a desired range and get a predicted launch angle
- Plot the experimental data together with the fitted model curves
"""

import numpy as np
import matplotlib.pyplot as plt


def fit_models():
    """Fit cubic polynomials for angle→range and range→angle."""
    # Experimental data
    angles_deg = np.array(
        [30, 45, 60, 75, 30, 45, 60, 75, 30, 80, 60, 75, 80],
        dtype=float,
    )
    ranges_m = np.array(
        [23.1, 27.8, 20.4, 14.2, 15.6, 18.1, 22.7, 16.5, 11.9, 8.1, 11.5, 8.5, 3.9],
        dtype=float,
    )

    # Model 1: range as a function of angle  (for predicting range at a given angle)
    coeff_angle_to_range = np.polyfit(angles_deg, ranges_m, deg=3)
    poly_angle_to_range = np.poly1d(coeff_angle_to_range)

    # Model 2: angle as a function of range  (for predicting angle given required range)
    coeff_range_to_angle = np.polyfit(ranges_m, angles_deg, deg=3)
    poly_range_to_angle = np.poly1d(coeff_range_to_angle)

    return (
        angles_deg,
        ranges_m,
        poly_angle_to_range,
        poly_range_to_angle,
    )


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute coefficient of determination R^2."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def main():
    (
        angles_deg,
        ranges_m,
        poly_angle_to_range,
        poly_range_to_angle,
    ) = fit_models()

    # a) Predict the range for angle = 50 degrees
    angle_test = 50.0
    predicted_range_50 = float(poly_angle_to_range(angle_test))

    # Compute R^2 for both directions to judge model accuracy
    r2_angle_to_range = r2_score(ranges_m, poly_angle_to_range(angles_deg))
    r2_range_to_angle = r2_score(angles_deg, poly_range_to_angle(ranges_m))

    print("=== Polynomial models (cubic) ===")
    print(f"Predicted range for angle = {angle_test:.1f}°: {predicted_range_50:.3f} m")
    print(f"R^2 (range as function of angle): {r2_angle_to_range:.3f}")
    print(f"R^2 (angle as function of range): {r2_range_to_angle:.3f}")

    # Predict inclination angle for a user-specified required range
    try:
        user_range_str = input(
            "Enter required range (in meters) to predict inclination angle (or press Enter to skip): "
        ).strip()
        if user_range_str:
            required_range = float(user_range_str)
            predicted_angle = float(poly_range_to_angle(required_range))
            print(
                f"Predicted inclination angle for range = {required_range:.3f} m: "
                f"{predicted_angle:.3f}°"
            )
    except ValueError:
        print("Could not parse the required range. Skipping angle prediction.")

    # b) Plot experimental data and model curves
    angle_grid = np.linspace(min(angles_deg) - 5, max(angles_deg) + 5, 400)
    range_pred_from_angle = poly_angle_to_range(angle_grid)

    range_grid = np.linspace(min(ranges_m), max(ranges_m), 400)
    angle_pred_from_range = poly_range_to_angle(range_grid)

    plt.figure(figsize=(10, 4.5))

    # Left: range vs angle (data + model)
    plt.subplot(1, 2, 1)
    plt.scatter(angles_deg, ranges_m, color="blue", label="Data", zorder=3)
    plt.plot(angle_grid, range_pred_from_angle, color="red", label="Model (angle→range)")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Range (m)")
    plt.title("Range vs Angle")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    # Right: angle vs range (data + model)
    plt.subplot(1, 2, 2)
    plt.scatter(ranges_m, angles_deg, color="green", label="Data", zorder=3)
    plt.plot(range_grid, angle_pred_from_range, color="orange", label="Model (range→angle)")
    plt.xlabel("Range (m)")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle vs Range")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # c) Very brief comment on accuracy
    print("\n=== Model accuracy comment ===")
    print(
        "The R^2 values above indicate how much of the variation in the data "
        "is captured by the cubic polynomial models. Values closer to 1 "
        "mean a more accurate model. Visually inspect the plots to see where "
        "the model deviates from the experimental points."
    )


if __name__ == "__main__":
    main()

