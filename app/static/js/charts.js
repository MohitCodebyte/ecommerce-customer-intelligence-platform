/**
 * CIX Platform — Chart.js Enterprise Formatting & Visualization System
 * Clean legends, short segment labels, full tooltip context, dynamic light/dark palette
 */

// =========================================================
// SEGMENT LABEL MAPPING
// =========================================================

const SEGMENT_LABEL_MAP = {
  'Champions': 'Champions',
  'Loyal Customers': 'Loyal',
  'Potential Loyalists': 'Potential',
  'At Risk High Value': 'High-Value Risk',
  'At Risk Valuable': 'Value Risk',
  'Needs Attention': 'Attention',
  'New / Promising': 'New',
  'Hibernating': 'Hibernating',
  'Very High': 'Very High Tier',
  'High': 'High Tier',
  'Medium': 'Medium Tier',
  'Low': 'Low Tier',
  'Below Average': 'Below Avg'
};

function getShortSegmentLabel(label) {
  if (!label) return '';
  return SEGMENT_LABEL_MAP[label] || label;
}


// =========================================================
// DESIGN TOKENS
// =========================================================

const DARK_COLORS = {
  primary: '#4cd7f6',
  secondary: '#c0c1ff',
  tertiary: '#4edea3',
  error: '#ffb4ab',
  warning: '#e8c84e',
  surface: '#1d1f27',
  onSurface: '#e1e2ec',
  onSurfaceVar: '#bcc9cd',
  grid: 'rgba(61,73,76,0.4)',
  tooltipBg: '#1d1f27',
  tooltipBorder: '#3d494c',
};

const LIGHT_COLORS = {
  primary: '#0284c7',
  secondary: '#6366f1',
  tertiary: '#059669',
  error: '#dc2626',
  warning: '#d97706',
  surface: '#ffffff',
  onSurface: '#0f172a',
  onSurfaceVar: '#475569',
  grid: 'rgba(226,232,240,0.8)',
  tooltipBg: '#ffffff',
  tooltipBorder: '#cbd5e1',
};


// =========================================================
// CHART PALETTES
// =========================================================

const DARK_PALETTE = [
  '#4cd7f6',
  '#c0c1ff',
  '#4edea3',
  '#ffb4ab',
  '#e8c84e',
  '#06b6d4',
  '#818cf8',
  '#34d399',
  '#f87171',
  '#fbbf24',
];

const LIGHT_PALETTE = [
  '#0284c7',
  '#6366f1',
  '#059669',
  '#dc2626',
  '#d97706',
  '#0891b2',
  '#4f46e5',
  '#10b981',
  '#ef4444',
  '#f59e0b',
];


// =========================================================
// THEME HELPERS
// =========================================================

function isDarkMode() {
  return document.documentElement.classList.contains('dark');
}

function getColors() {
  return isDarkMode()
    ? DARK_COLORS
    : LIGHT_COLORS;
}

function getPalette() {
  return isDarkMode()
    ? DARK_PALETTE
    : LIGHT_PALETTE;
}


// =========================================================
// GLOBAL CHART FONT
// =========================================================

const BASE_FONT = {
  family: 'Inter, sans-serif',
  size: 11,
};

Chart.defaults.font.family = BASE_FONT.family;
Chart.defaults.font.size = BASE_FONT.size;


// =========================================================
// FORMAT HELPERS
// =========================================================

function fmtNum(n) {
  if (n == null) return '—';

  if (n >= 1e6) {
    return (n / 1e6).toFixed(1) + 'M';
  }

  if (n >= 1e3) {
    return (n / 1e3).toFixed(1) + 'K';
  }

  return typeof n === 'number'
    ? n.toLocaleString()
    : n;
}


function fmtCurrency(n) {
  if (n == null) return '—';

  return '$' + fmtNum(n);
}


function fmtPct(n) {
  if (n == null) return '—';

  return (n * 100).toFixed(1) + '%';
}


// =========================================================
// DONUT / DOUGHNUT CHART
// =========================================================

function createDonutChart(
  canvasId,
  rawLabels,
  data,
  title = ''
) {

  const ctx =
    document.getElementById(canvasId);

  if (!ctx) return null;

  const colors = getColors();
  const palette = getPalette();

  const shortLabels =
    rawLabels.map(getShortSegmentLabel);

  const total = data.reduce(
    (a, b) => a + (b || 0),
    0
  );


  // =======================================================
  // LEGEND TEXT COLOR
  // Dark  = White
  // Light = Dark
  // =======================================================

  const legendTextColor = isDarkMode()
    ? '#ffffff'
    : '#1e293b';


  return new Chart(ctx, {

    type: 'doughnut',

    data: {

      labels: shortLabels,

      datasets: [
        {
          data,

          backgroundColor:
            palette
              .slice(0, data.length)
              .map(c => c + 'cc'),

          borderColor:
            palette
              .slice(0, data.length),

          borderWidth: 1.5,

          hoverOffset: 6,
        }
      ]
    },


    options: {

      responsive: true,

      maintainAspectRatio: false,

      cutout: '68%',


      plugins: {

        // =================================================
        // LEGEND
        // =================================================

        legend: {

          position:
            window.innerWidth < 768
              ? 'bottom'
              : 'right',

          labels: {

            // Main legend color
            color: legendTextColor,

            boxWidth: 10,

            boxHeight: 10,

            padding: 10,

            font: {
              size: 11,
              family: 'Inter, sans-serif',
              weight: '500',
            },


            // =================================================
            // CUSTOM LEGEND GENERATION
            // IMPORTANT:
            // fontColor MUST be included in each item.
            // Otherwise Chart.js canvas defaults to BLACK.
            // =================================================

            generateLabels: (chart) => {

              const datasets =
                chart.data.datasets;

              return chart.data.labels.map(
                (lbl, i) => {

                  const val =
                    datasets[0].data[i] || 0;


                  const pct =
                    total > 0
                      ? (
                          (val / total) * 100
                        ).toFixed(1) + '%'
                      : '';


                  return {

                    text:
                      `${lbl} (${pct})`,

                    fillStyle:
                      datasets[0]
                        .backgroundColor[i],

                    strokeStyle:
                      datasets[0]
                        .borderColor[i],

                    lineWidth: 1.5,

                    hidden:
                      !chart.getDataVisibility(i),

                    index: i,


                    // =================================================
                    // THIS IS THE IMPORTANT FIX
                    // =================================================

                    fontColor:
                      legendTextColor,
                  };

                }
              );
            }
          }
        },


        // =================================================
        // TOOLTIP
        // =================================================

        tooltip: {

          backgroundColor:
            colors.tooltipBg,

          titleColor:
            colors.onSurface,

          bodyColor:
            colors.onSurfaceVar,

          borderColor:
            colors.tooltipBorder,

          borderWidth: 1,

          padding: 10,

          cornerRadius: 8,


          callbacks: {

            title: (items) => {

              const idx =
                items[0].dataIndex;

              return (
                rawLabels[idx] ||
                shortLabels[idx]
              );
            },


            label: (item) => {

              const val =
                item.raw || 0;

              const pct =
                total > 0
                  ? (
                      (val / total) * 100
                    ).toFixed(1)
                  : 0;

              return ` Count: ${val.toLocaleString()} (${pct}%)`;
            }
          }
        },


        // =================================================
        // OPTIONAL TITLE
        // =================================================

        title: title

          ? {

              display: true,

              text: title,

              color:
                colors.onSurface,

              font: {
                size: 13,
                weight: '600',
              },

            }

          : {
              display: false,
            }
      }
    }
  });
}


// =========================================================
// BAR CHART
// =========================================================

function createBarChart(
  canvasId,
  rawLabels,
  datasets,
  options = {}
) {

  const ctx =
    document.getElementById(canvasId);

  if (!ctx) return null;

  const colors = getColors();

  const palette = getPalette();

  const shortLabels =
    rawLabels.map(getShortSegmentLabel);


  return new Chart(ctx, {

    type: 'bar',

    data: {

      labels: shortLabels,

      datasets:

        datasets.map((d, i) => ({

          backgroundColor:
            (palette[i] || colors.primary) +
            '88',

          borderColor:
            palette[i] ||
            colors.primary,

          hoverBackgroundColor:
            palette[i] ||
            colors.primary,

          borderWidth: 1.5,

          borderRadius: 4,

          ...d,
        }))
    },


    options: {

      responsive: true,

      maintainAspectRatio: false,


      plugins: {

        legend: {

          display:
            datasets.length > 1,

          labels: {

            color:
              isDarkMode()
                ? '#ffffff'
                : '#1e293b',

            boxWidth: 10,
          }
        },


        tooltip: {

          backgroundColor:
            colors.tooltipBg,

          titleColor:
            colors.onSurface,

          bodyColor:
            colors.onSurfaceVar,

          borderColor:
            colors.tooltipBorder,

          borderWidth: 1,

          cornerRadius: 8,


          callbacks: {

            title: (items) =>

              rawLabels[
                items[0].dataIndex
              ] ||

              shortLabels[
                items[0].dataIndex
              ]
          }
        }
      },


      scales: {

        x: {

          grid: {

            color:
              colors.grid,

            drawBorder: false,
          },


          ticks: {

            color:
              colors.onSurfaceVar,

            font: {
              size: 11,
            }
          },


          ...options.x,
        },


        y: {

          grid: {

            color:
              colors.grid,

            drawBorder: false,
          },


          ticks: {

            color:
              colors.onSurfaceVar,

            font: {
              size: 11,
            }
          },


          beginAtZero: true,

          ...options.y,
        }
      },


      ...options.chart,
    }
  });
}


// =========================================================
// HORIZONTAL BAR CHART
// =========================================================

function createHorizontalBarChart(
  canvasId,
  rawLabels,
  data,
  color
) {

  const ctx =
    document.getElementById(canvasId);

  if (!ctx) return null;

  const colors = getColors();

  const barColor =
    color ||
    colors.primary;

  const shortLabels =
    rawLabels.map(getShortSegmentLabel);


  return new Chart(ctx, {

    type: 'bar',

    data: {

      labels: shortLabels,

      datasets: [

        {

          data,

          backgroundColor:
            barColor + '88',

          borderColor:
            barColor,

          borderWidth: 1.5,

          borderRadius: 4,

          hoverBackgroundColor:
            barColor,
        }
      ]
    },


    options: {

      indexAxis: 'y',

      responsive: true,

      maintainAspectRatio: false,


      plugins: {

        legend: {

          display: false,
        },


        tooltip: {

          backgroundColor:
            colors.tooltipBg,

          titleColor:
            colors.onSurface,

          bodyColor:
            colors.onSurfaceVar,

          borderColor:
            colors.tooltipBorder,

          borderWidth: 1,

          cornerRadius: 8,


          callbacks: {

            title: (items) =>

              rawLabels[
                items[0].dataIndex
              ] ||

              shortLabels[
                items[0].dataIndex
              ]
          }
        }
      },


      scales: {

        x: {

          grid: {

            color:
              colors.grid,

            drawBorder: false,
          },


          ticks: {

            color:
              colors.onSurfaceVar,
          },


          beginAtZero: true,
        },


        y: {

          grid: {

            display: false,
          },


          ticks: {

            color:
              colors.onSurfaceVar,

            font: {
              size: 11,
            }
          }
        }
      }
    }
  });
}


// =========================================================
// LINE CHART
// =========================================================

function createLineChart(
  canvasId,
  rawLabels,
  datasets,
  options = {}
) {

  const ctx =
    document.getElementById(canvasId);

  if (!ctx) return null;

  const colors = getColors();

  const palette = getPalette();

  const shortLabels =
    rawLabels.map(getShortSegmentLabel);


  return new Chart(ctx, {

    type: 'line',

    data: {

      labels: shortLabels,

      datasets:

        datasets.map((d, i) => ({

          borderColor:
            palette[i] ||
            colors.primary,

          backgroundColor:
            (
              palette[i] ||
              colors.primary
            ) + '18',

          borderWidth: 2,

          fill: true,

          tension: 0.4,

          pointBackgroundColor:
            palette[i] ||
            colors.primary,

          pointRadius: 3,

          pointHoverRadius: 6,

          ...d,
        }))
    },


    options: {

      responsive: true,

      maintainAspectRatio: false,


      plugins: {

        legend: {

          labels: {

            color:
              isDarkMode()
                ? '#ffffff'
                : '#1e293b',

            boxWidth: 10,
          }
        },


        tooltip: {

          backgroundColor:
            colors.tooltipBg,

          titleColor:
            colors.onSurface,

          bodyColor:
            colors.onSurfaceVar,

          borderColor:
            colors.tooltipBorder,

          borderWidth: 1,

          cornerRadius: 8,

          mode: 'index',

          intersect: false,
        }
      },


      scales: {

        x: {

          grid: {

            color:
              colors.grid,

            drawBorder: false,
          },


          ticks: {

            color:
              colors.onSurfaceVar,
          },


          ...options.x,
        },


        y: {

          grid: {

            color:
              colors.grid,

            drawBorder: false,
          },


          ticks: {

            color:
              colors.onSurfaceVar,
          },


          beginAtZero: true,

          ...options.y,
        }
      },


      interaction: {

        mode: 'index',

        intersect: false,
      },


      ...options.chart,
    }
  });
}