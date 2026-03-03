// Experimental data (same as in the Python script)
const anglesDeg = [30, 45, 60, 75, 30, 45, 60, 75, 30, 80, 60, 75, 80];
const rangesM = [23.1, 27.8, 20.4, 14.2, 15.6, 18.1, 22.7, 16.5, 11.9, 8.1, 11.5, 8.5, 3.9];

// Simple helper functions to compute polynomial regression and R^2 in JS
function polyfit(x, y, degree) {
  // Build Vandermonde matrix
  const n = x.length;
  const X = new Array(degree + 1).fill(0).map(() => new Array(degree + 1).fill(0));
  const Y = new Array(degree + 1).fill(0);

  for (let row = 0; row <= degree; row++) {
    for (let col = 0; col <= degree; col++) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        sum += Math.pow(x[i], row + col);
      }
      X[row][col] = sum;
    }
    let sumY = 0;
    for (let i = 0; i < n; i++) {
      sumY += y[i] * Math.pow(x[i], row);
    }
    Y[row] = sumY;
  }

  const coeffs = solveLinearSystem(X, Y); // returns [c0, c1, ..., c_degree]
  // Convert to highest-degree-first order like numpy.poly1d
  return coeffs.reverse();
}

function solveLinearSystem(A, b) {
  // Gaussian elimination (small system, fine for this case)
  const n = A.length;
  const M = A.map((row, i) => row.concat(b[i]));

  for (let k = 0; k < n; k++) {
    // Pivot
    let maxRow = k;
    for (let i = k + 1; i < n; i++) {
      if (Math.abs(M[i][k]) > Math.abs(M[maxRow][k])) maxRow = i;
    }
    [M[k], M[maxRow]] = [M[maxRow], M[k]];

    const pivot = M[k][k];
    if (Math.abs(pivot) < 1e-12) continue;

    // Normalize pivot row
    for (let j = k; j <= n; j++) {
      M[k][j] /= pivot;
    }

    // Eliminate
    for (let i = 0; i < n; i++) {
      if (i === k) continue;
      const factor = M[i][k];
      for (let j = k; j <= n; j++) {
        M[i][j] -= factor * M[k][j];
      }
    }
  }

  return M.map((row) => row[n]);
}

function polyval(coeffs, x) {
  // coeffs: [a3, a2, a1, a0]
  return coeffs.reduce((acc, c) => acc * x + c, 0);
}

function r2Score(yTrue, yPred) {
  const n = yTrue.length;
  const mean = yTrue.reduce((a, v) => a + v, 0) / n;
  let ssRes = 0;
  let ssTot = 0;
  for (let i = 0; i < n; i++) {
    ssRes += Math.pow(yTrue[i] - yPred[i], 2);
    ssTot += Math.pow(yTrue[i] - mean, 2);
  }
  return 1 - ssRes / ssTot;
}

// Fit cubic models
const coeffAngleToRange = polyfit(anglesDeg, rangesM, 3); // angle -> range
const coeffRangeToAngle = polyfit(rangesM, anglesDeg, 3); // range -> angle

// Precompute R^2 values
const rangesPredFromAngles = anglesDeg.map((a) => polyval(coeffAngleToRange, a));
const anglesPredFromRanges = rangesM.map((r) => polyval(coeffRangeToAngle, r));
const r2AngleToRange = r2Score(rangesM, rangesPredFromAngles);
const r2RangeToAngle = r2Score(anglesDeg, anglesPredFromRanges);

// DOM elements
const angleInput = document.getElementById("angleInput");
const angleValue = document.getElementById("angleValue");
const rangePrediction = document.getElementById("rangePrediction");
const angleToRangeBtn = document.getElementById("angleToRangeBtn");

const rangeInput = document.getElementById("rangeInput");
const requiredRangeLabel = document.getElementById("requiredRangeLabel");
const anglePrediction = document.getElementById("anglePrediction");
const rangeToAngleBtn = document.getElementById("rangeToAngleBtn");

const r2AngleToRangeSpan = document.getElementById("r2AngleToRange");
const r2RangeToAngleSpan = document.getElementById("r2RangeToAngle");

// Initialize R^2 display
r2AngleToRangeSpan.textContent = r2AngleToRange.toFixed(3);
r2RangeToAngleSpan.textContent = r2RangeToAngle.toFixed(3);

// Prediction functions
function updateAngleToRange() {
  const angle = parseFloat(angleInput.value);
  angleValue.textContent = `${angle.toFixed(0)}°`;
  const predRange = polyval(coeffAngleToRange, angle);
  rangePrediction.textContent = `Range ≈ ${predRange.toFixed(2)} m`;
}

function updateRangeToAngle() {
  const val = parseFloat(rangeInput.value);
  if (Number.isNaN(val)) {
    anglePrediction.textContent = "Angle ≈  °";
    requiredRangeLabel.textContent = " m";
    return;
  }
  requiredRangeLabel.textContent = `${val.toFixed(1)} m`;
  const predAngle = polyval(coeffRangeToAngle, val);
  anglePrediction.textContent = `Angle ≈ ${predAngle.toFixed(2)}°`;
}

angleInput.addEventListener("input", updateAngleToRange);
angleToRangeBtn.addEventListener("click", updateAngleToRange);
rangeToAngleBtn.addEventListener("click", updateRangeToAngle);
rangeInput.addEventListener("change", updateRangeToAngle);
rangeInput.addEventListener("input", updateRangeToAngle);

// Initial predictions
updateAngleToRange();
updateRangeToAngle();

// Charts
const angleRangeCtx = document.getElementById("angleRangeChart").getContext("2d");
const rangeAngleCtx = document.getElementById("rangeAngleChart").getContext("2d");

// Smooth curves
const angleGrid = [];
for (let a = 20; a <= 85; a += 0.5) {
  angleGrid.push(a);
}
const rangeCurve = angleGrid.map((a) => polyval(coeffAngleToRange, a));

const minRange = Math.min(...rangesM);
const maxRange = Math.max(...rangesM);
const rangeGrid = [];
for (let r = minRange; r <= maxRange; r += (maxRange - minRange) / 200) {
  rangeGrid.push(r);
}
const angleCurve = rangeGrid.map((r) => polyval(coeffRangeToAngle, r));

new Chart(angleRangeCtx, {
  type: "scatter",
  data: {
    datasets: [
      {
        label: "Experimental data",
        data: anglesDeg.map((a, i) => ({ x: a, y: rangesM[i] })),
        backgroundColor: "#3b82f6",
        pointRadius: 4,
      },
      {
        type: "line",
        label: "Model (angle → range)",
        data: angleGrid.map((a, i) => ({ x: a, y: rangeCurve[i] })),
        borderColor: "#f97316",
        borderWidth: 2,
        pointRadius: 0,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: { display: true, text: "Angle (degrees)" },
        grid: { color: "rgba(148, 163, 184, 0.2)" },
      },
      y: {
        title: { display: true, text: "Range (m)" },
        grid: { color: "rgba(148, 163, 184, 0.2)" },
      },
    },
    plugins: {
      legend: {
        labels: { color: "#e5e7eb" },
      },
    },
  },
});

new Chart(rangeAngleCtx, {
  type: "scatter",
  data: {
    datasets: [
      {
        label: "Experimental data",
        data: rangesM.map((r, i) => ({ x: r, y: anglesDeg[i] })),
        backgroundColor: "#22c55e",
        pointRadius: 4,
      },
      {
        type: "line",
        label: "Model (range → angle)",
        data: rangeGrid.map((r, i) => ({ x: r, y: angleCurve[i] })),
        borderColor: "#eab308",
        borderWidth: 2,
        pointRadius: 0,
      },
    ],
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        title: { display: true, text: "Range (m)" },
        grid: { color: "rgba(148, 163, 184, 0.2)" },
      },
      y: {
        title: { display: true, text: "Angle (degrees)" },
        grid: { color: "rgba(148, 163, 184, 0.2)" },
      },
    },
    plugins: {
      legend: {
        labels: { color: "#e5e7eb" },
      },
    },
  },
});

